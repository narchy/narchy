package jcog.tensor.model;

import jcog.tensor.Tensor;
import jcog.util.ArrayUtil;

import static jcog.Util.clampSafe;
import static jcog.Util.unitize;

/**
 * Revised DNCMemory with:
 * 1. Optional erase (controlled via the useErase flag).
 * 2. Elimination of queryStrengths from the controller output when reads==1.
 */
public class DNCMemory {
    public final int size, width, reads;
    public final EraseMode eraseMode;

    // Memory tensor: [size, width]
    public final Tensor memory;
    // Read weights: [reads, size]
    public final Tensor read;
    // Write weights: [1, size]
    public final Tensor write;
    // Usage: [1, size]
    public final Tensor use;

    public final MemoryAddressing addressing;
    public final RetentionStrategy retention;

    /** Memory clamping range */
    private static final double memoryMin = -2, memoryMax = +2;

    /**
     * Erase modes:
     *   FULL:      Controller outputs a full erase vector.
     *   CONSTANT:  Controller outputs one scalar (to be broadcast).
     *   DISABLED:  No erase is applied.
     */
    public enum EraseMode {
        FULL,
        SCALAR,
        DISABLED
    }

    /**
     * Constructor with explicit erase mode.
     */
    public DNCMemory(int size, int width, int reads,
                     MemoryAddressing addressing,
                     RetentionStrategy retention,
                     EraseMode eraseMode) {
        this.size = size;
        this.width = width;
        this.reads = reads;
        this.eraseMode = eraseMode;

        this.memory = Tensor.zeros(size, width).grad(true);
        this.read = Tensor.zeros(reads, size).grad(true);
        this.write = Tensor.zeros(1, size).grad(true);
        this.use = Tensor.zeros(1, size).grad(true);

        this.addressing = addressing;
        this.retention = retention;

        reset();
    }

    /**
     * Convenience constructor (defaults to FULL erase mode).
     */
    public DNCMemory(int size, int width, int reads,
                     MemoryAddressing addressing,
                     RetentionStrategy retention) {
        this(size, width, reads, addressing, retention, EraseMode.FULL);
    }

    public void reset() {
        memory.zero();
        read.fill(1.0 / size);
        write.fill(1.0 / size);
        use.zero();
    }

    /**
     * Computes the total number of scalars expected from the controller for memory control.
     */
    public int controlSize() {
        int cs = width; // writeVector always
        // Erase signals:
        if (eraseMode == EraseMode.FULL) {
            cs += width;
        } else if (eraseMode == EraseMode.SCALAR) {
            cs += 1;
        }
        // Query keys: always reads*width
        cs += (reads * width);
        // Query strengths: only if more than one read head (otherwise use constant 1.0)
        if (reads > 1) {
            cs += reads;
        }
        // Write weight: always size
        cs += size;
        return cs;
    }

    /**
     * Reads from memory using the current read weights.
     * Returns a tensor of shape [reads, width].
     */
    public Tensor read() {
        return read.matmul(memory);
    }

    /**
     * Write operation:
     *   memory = memory * (1 - (write^T . eraseVector)) + (write^T . writeVector)
     */
    private void write(Tensor eraseVector, Tensor writeVector) {
        retention.updateUsage(use, write);
        Tensor wErase = write.transposeMatmul(eraseVector); // [size, width]
        Tensor wWrite = write.transposeMatmul(writeVector);   // [size, width]
        double[] m = memory.array(), e = wErase.array(), w = wWrite.array();
        for (int i = 0, length = m.length; i < length; i++) {
            m[i] = clampSafe(m[i] * (1 - e[i]) + w[i], memoryMin, memoryMax);
        }
    }

    /**
     * Update read weights using content-based addressing.
     */
    private void updateQuery(Tensor queryKeys, Tensor queryStrengths) {
        addressing.updateReadWeights(memory, read, queryKeys, queryStrengths);
    }

    /**
     * Full memory update:
     *  - Update the write weight.
     *  - Update read weights using query keys and (if applicable) query strengths.
     *  - Write to memory using the erase and write vectors.
     */
    public void update(ControlSignals controls) {
        updateWriteWeight(controls.writeWeight);
        updateQuery(controls.queryKeys, controls.queryStrengths);
        write(controls.eraseVector, controls.writeVector);
    }

    public void updateWriteWeight(Tensor writeWeight) {
        write.setData(writeWeight.array());
    }

    public int stateSize() {
        return reads * width;
    }

    /**
     * Extracts control signals from the controller output.
     *
     * Layout:
     *   1. writeVector: [1, width]
     *   2. Erase:
     *       - FULL:     [1, width]
     *       - CONSTANT: [1, 1] (to be broadcast)
     *       - DISABLED: none (a zero vector is used)
     *   3. queryKeys: [reads, width]
     *   4. Query strengths: if reads > 1 then [1, reads], else (constant 1.0)
     *   5. writeWeight: [1, size]
     */
    public ControlSignals extractControlSignals(Tensor controlSlice) {
        double[] data = controlSlice.array();
        int offset = 0;
        // 1) writeVector: [1, width]
        int wSize = width;
        double[] wArr = new double[wSize];
        System.arraycopy(data, offset, wArr, 0, wSize);
        offset += wSize;
        Tensor writeVector = Tensor.row(wArr);

        // 2) Erase signal:
        Tensor eraseVector;
        if (eraseMode == EraseMode.FULL) {
            double[] eArr = new double[width];
            System.arraycopy(data, offset, eArr, 0, width);
            offset += width;
            eraseVector = Tensor.row(eArr);
        } else if (eraseMode == EraseMode.SCALAR) {
            double[] cArr = new double[1];
            System.arraycopy(data, offset, cArr, 0, 1);
            offset += 1;
            // Broadcast the constant value to a full vector
            eraseVector = Tensor.fill(1, width, cArr[0]);
        } else { // DISABLED: use a zero vector.
            eraseVector = Tensor.zeros(1, width);
        }

        // 3) Query keys: [reads, width]
        int kSize = reads * width;
        double[] kArr = new double[kSize];
        System.arraycopy(data, offset, kArr, 0, kSize);
        offset += kSize;
        Tensor queryKeys = new Tensor(kArr, reads, width, false);

        // 4) Query strengths:
        Tensor queryStrengths;
        if (reads > 1) {
            int sSize = reads;
            double[] sArr = new double[sSize];
            System.arraycopy(data, offset, sArr, 0, sSize);
            offset += sSize;
            queryStrengths = Tensor.row(sArr);
        } else {
            // If there's only one read head, use a fixed value of 1.0.
            queryStrengths = Tensor.scalar(1);
        }

        // 5) Write weight: [1, size]
        int dSize = size;
        double[] dArr = new double[dSize];
        System.arraycopy(data, offset, dArr, 0, dSize);
        offset += dSize;
        Tensor writeWeight = Tensor.row(dArr).softmax();

        if (offset != controlSlice.volume()) throw new jcog.WTF();

        return new ControlSignals(writeVector, eraseVector, queryKeys, queryStrengths, writeWeight);
    }

    /**
     * Represents the memory control parameters.
     */
    public static class ControlSignals {
        public final Tensor writeVector;    // [1, width]
        public final Tensor eraseVector;    // [1, width]
        public final Tensor queryKeys;      // [reads, width]
        public final Tensor queryStrengths; // [1, (reads>1 ? reads : 1)]
        public final Tensor writeWeight;    // [1, size]

        public ControlSignals(Tensor writeVector,
                              Tensor eraseVector,
                              Tensor queryKeys,
                              Tensor queryStrengths,
                              Tensor writeWeight) {
            this.writeVector = writeVector.clipData(-1, +1);
            this.eraseVector = eraseVector.clipData(0, +1);
            this.queryKeys = queryKeys.clipData(-1, +1);
            this.writeWeight = writeWeight; // Already softmaxed.
            this.queryStrengths = queryStrengths.softplus();
        }
    }

    /**
     * Memory addressing interface (content-based).
     */
    public interface MemoryAddressing {
        void updateReadWeights(Tensor memory,
                               Tensor readWeights,
                               Tensor queryKeys,
                               Tensor queryStrengths);
    }

    /**
     * Retention strategy interface.
     */
    public interface RetentionStrategy {
        void updateUsage(Tensor usage, Tensor writeWeights);
        default int[] getFreeLocations(Tensor usage, int count) {
            return ArrayUtil.EMPTY_INT_ARRAY;
        }
    }

    /**
     * Computes cosine similarity between each memory row and a key (row vector).
     * Returns a tensor of shape [1, size].
     */
    static Tensor cosineSimilarityRow(Tensor memory, Tensor keyRow) {
        double eps = 1e-8;
        Tensor dotVec = memory.matmulTranspose(keyRow); // [size, 1]
        Tensor memNorm = memory.rowNormsL2();             // [size, 1]
        var keyNorm = keyRow.lenL2();
        Tensor denom = memNorm.mul(keyNorm).add(eps);
        Tensor sim = dotVec.div(denom);
        return sim.transpose(); // [1, size]
    }

    /**
     * Content-based addressing using cosine similarity and softmax.
     */
    public static class ContentBasedAddressing implements MemoryAddressing {
        @Override
        public void updateReadWeights(Tensor memory,
                                      Tensor readWeights,
                                      Tensor queryKeys,
                                      Tensor queryStrengths) {
            if (queryKeys.rows() == 1) {
                double strength = queryStrengths.scalar();
                Tensor simRow = cosineSimilarityRow(memory, queryKeys).mul(strength);
                Tensor sm = simRow.softmax();
                readWeights.setDataRow(0, sm);
            } else {
                int R = queryKeys.rows(), C = queryKeys.cols();
                for (int r = 0; r < R; r++) {
                    Tensor qKeyRow = queryKeys.slice(r, r + 1, 0, C);
                    double strength = queryStrengths.data(r);
                    Tensor simRow = cosineSimilarityRow(memory, qKeyRow).mul(strength);
                    Tensor sm = simRow.softmax();
                    readWeights.setDataRow(r, sm);
                }
            }
        }
    }

    /**
     * Simple usage-based retention: usage = clamp(usage + writeWeights, 0, 1).
     */
    public static class UsageBasedRetention implements RetentionStrategy {
        @Override
        public void updateUsage(Tensor usage, Tensor writeWeights) {
            double[] u = usage.array(), w = writeWeights.array();
            for (int i = 0, length = u.length; i < length; i++) {
                u[i] = unitize(u[i] + w[i]);
            }
        }
    }
}