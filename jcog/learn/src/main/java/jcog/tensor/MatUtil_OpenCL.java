package jcog.tensor;

import com.jogamp.opencl.*;
import jcog.Log;
import jcog.Util;
import org.ejml.data.DMatrixRMaj;
import org.slf4j.Logger;

import java.lang.AutoCloseable;
import java.nio.DoubleBuffer;

public class MatUtil_OpenCL implements AutoCloseable {

    public static final int ELEMENT_THRESHOLD = 1024*1024;

    private final static Logger logger = Log.log(MatUtil_OpenCL.class);
    private static final String matrixMultiply = """
            __kernel void matrixMultiply(
                const int M, const int N, const int K,
                const __global double* A,
                const __global double* B,
                      __global double* C) {
            
                const int row = get_global_id(0), col = get_global_id(1);            
                if (row < M && col < N) {
                    double sum = 0.0;
                    for (int k = 0; k < K; k++)
                        sum += A[row * K + k] * B[k * N + col];               
                    C[row * N + col] = sum;
                }
            }
            """;
    private final CLContext cl = CLContext.create();

    private final CLCommandQueue queue;
    private final CLKernel mm;
    private CLBuffer<DoubleBuffer> reusableBuffer = null;
    private int reusableBufferSize = 0;

    public MatUtil_OpenCL() {
        var device = cl.getMaxFlopsDevice();
        //queue = device.createCommandQueue();
        queue = device.createCommandQueue(
                CLCommandQueue.Mode.OUT_OF_ORDER_MODE
                //CLCommandQueue.Mode.PROFILING_MODE,
        );


        mm = cl.createProgram(matrixMultiply).build().createCLKernel("matrixMultiply");
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (!cl.isReleased())
                close();
        }));
    }

    public static void main(String[] args) {

        try (var multiplier = new MatUtil_OpenCL()) {
            for (int d : new int[]{1, 1, 1, /* warm up */ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
                int x = d, y = d, z = d;
                Util.time("mult_" + d, logger, () -> {
                    multiplier.mult(
                            new double[x * y], new double[y * z], new double[x * z],
                            x, y, z);
                });
            }
        }

    }

    private final static ThreadLocal<MatUtil_OpenCL> clMatMul = ThreadLocal.withInitial(MatUtil_OpenCL::new);

    public static MatUtil_OpenCL get() {
        return clMatMul.get();
    }

    public void mult(DMatrixRMaj a, DMatrixRMaj b, DMatrixRMaj c) {
        int x = a.getNumRows(), y = a.getNumCols(), z = b.getNumCols();
        mult(a.data, b.data, c.data, x, y, z);
    }

    /**
     * C[x,z] = A[x,y] * B[y,z]
     */
    public void mult(double[] A, double[] B, double[] C, int x, int y, int z) {

        int requiredSizeA = x * y, requiredSizeB = y * z, requiredSizeC = x * z;

        // Ensure buffer is large enough for A, B, and C
        var buffer = ensureBufferCapacity(requiredSizeA + requiredSizeB + requiredSizeC);

        // Split the reusable buffer into sections for A, B, and C
        CLSubBuffer<DoubleBuffer> bufferA = buffer.createSubBuffer(0, requiredSizeA, CLBuffer.Mem.READ_ONLY);
        CLSubBuffer<DoubleBuffer> bufferB = buffer.createSubBuffer(requiredSizeA, requiredSizeB, CLBuffer.Mem.READ_ONLY);
        CLSubBuffer<DoubleBuffer> bufferC = buffer.createSubBuffer(requiredSizeA + requiredSizeB, requiredSizeC, CLBuffer.Mem.WRITE_ONLY);

        try {
            // Write A and B data into the reusable buffer
            bufferA.getBuffer().put(A).rewind();
            bufferB.getBuffer().put(B).rewind();

            mm.setArgs(x, y, z, bufferA, bufferB, bufferC);

            queue.putWriteBuffer(bufferA, false)
                    .putWriteBuffer(bufferB, false)
                    .put2DRangeKernel(mm, 0, 0, x, z, 0, 0)  // x and z define work size
                    .putReadBuffer(bufferC, true);

            queue.finish();

            bufferC.getBuffer().get(C); // Read back the result into C


        } finally {
            bufferA.release();
            bufferB.release();
            bufferC.release();
        }

    }

    /**
     * Ensure the reusable buffer is large enough for the current operation
     *
     * @return
     */
    private CLBuffer<DoubleBuffer> ensureBufferCapacity(int requiredSize) {
        if (reusableBuffer == null || reusableBufferSize < requiredSize) {
            if (reusableBuffer != null)
                reusableBuffer.release(); // Release the old buffer if it exists
            reusableBuffer = cl.createDoubleBuffer(requiredSize,
                    CLBuffer.Mem.READ_WRITE
                    ,
                    CLBuffer.Mem.ALLOCATE_BUFFER
                    //CLBuffer.Mem.USE_BUFFER
            );
            reusableBufferSize = requiredSize;
        }
        return reusableBuffer;
    }

    @Override
    public void close() {
        if (reusableBuffer != null && !reusableBuffer.isReleased())
            reusableBuffer.release();

        cl.release();
    }
}
//public class MatUtil_OpenCL implements AutoCloseable {
//
//    private static final Logger logger = Log.log(MatUtil_OpenCL.class);
//    private static final String KERNEL_NAME = "matrixMultiplyOptimized";
//
//    // OpenCL Kernel as a String (Single-Precision with Tiling)
//    private static final String matrixMultiplyOptimized = """
//        __kernel void matrixMultiplyOptimized(
//            const int M, const int N, const int K,
//            __global float* A,
//            __global float* B,
//            __global float* C) {
//
//            // Tile size (adjust based on device capabilities)
//            const int TILE_SIZE = 16;
//
//            // Local memory for tiles of A and B
//            __local float Asub[TILE_SIZE][TILE_SIZE];
//            __local float Bsub[TILE_SIZE][TILE_SIZE];
//
//            int row = get_global_id(0);
//            int col = get_global_id(1);
//            float sum = 0.0f;
//
//            // Loop over tiles
//            for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
//                // Load tiles into local memory
//                int tiledRow = row;
//                int tiledCol = t * TILE_SIZE + get_local_id(1);
//                Asub[get_local_id(0)][get_local_id(1)] = (tiledCol < K && tiledRow < M) ? A[tiledRow * K + tiledCol] : 0.0f;
//
//                tiledRow = t * TILE_SIZE + get_local_id(0);
//                tiledCol = col;
//                Bsub[get_local_id(0)][get_local_id(1)] = (tiledRow < K && tiledCol < N) ? B[tiledRow * N + tiledCol] : 0.0f;
//
//                barrier(CLK_LOCAL_MEM_FENCE);
//
//                // Compute partial sums
//                for (int k = 0; k < TILE_SIZE; k++) {
//                    sum += Asub[get_local_id(0)][k] * Bsub[k][get_local_id(1)];
//                }
//
//                barrier(CLK_LOCAL_MEM_FENCE);
//            }
//
//            // Write the result
//            if (row < M && col < N) {
//                C[row * N + col] = sum;
//            }
//        }
//        """;
//    public static int ELEMENT_THRESHOLD = 1024 * 1024;
//
//    private final CLContext cl;
//    private final CLCommandQueue queue;
//    private final CLKernel mm;
//
//    // Buffer pools for A, B, and C
//    private final ConcurrentLinkedQueue<CLBuffer<FloatBuffer>> bufferPoolA = new ConcurrentLinkedQueue<>();
//    private final ConcurrentLinkedQueue<CLBuffer<FloatBuffer>> bufferPoolB = new ConcurrentLinkedQueue<>();
//    private final ConcurrentLinkedQueue<CLBuffer<FloatBuffer>> bufferPoolC = new ConcurrentLinkedQueue<>();
//
//    /**
//     * Constructs a MatUtil_OpenCL instance, initializing OpenCL context, command queue, and kernel.
//     */
//    public MatUtil_OpenCL() {
//        cl = CLContext.create();
//        CLDevice device = cl.getMaxFlopsDevice();
//        queue = device.createCommandQueue(CLCommandQueue.Mode.OUT_OF_ORDER_MODE);
//
//        try {
//            CLProgram program = cl.createProgram(matrixMultiplyOptimized).build();
//            mm = program.createCLKernel(KERNEL_NAME);
//        } catch (CLException e) {
//            logger.error("Failed to build OpenCL program or create kernel", e);
//            throw e;
//        }
//
//        // Ensure resources are released on JVM shutdown
//        Runtime.getRuntime().addShutdownHook(new Thread(this::close));
//    }
//
//    /**
//     * ThreadLocal instance for thread-safe MatUtil_OpenCL usage.
//     */
//    private final static ThreadLocal<MatUtil_OpenCL> clMatMul = ThreadLocal.withInitial(MatUtil_OpenCL::new);
//
//    /**
//     * Retrieves the thread-local MatUtil_OpenCL instance.
//     *
//     * @return MatUtil_OpenCL instance
//     */
//    public static MatUtil_OpenCL get() {
//        return clMatMul.get();
//    }
//
//    /**
//     * Multiplies two matrices using EJML's DMatrixRMaj by converting them to float arrays.
//     *
//     * @param a Input matrix A
//     * @param b Input matrix B
//     * @param c Output matrix C (A * B)
//     */
//    public void mult(DMatrixRMaj a, DMatrixRMaj b, DMatrixRMaj c) {
//        int x = a.getNumRows(), y = a.getNumCols(), z = b.getNumCols();
//        if (y != b.getNumRows())
//            throw new IllegalArgumentException("Matrix dimensions do not match for multiplication.");
//
//        // Convert double arrays to float arrays
//        float[] A = convertToFloat(a.data);
//        float[] B = convertToFloat(b.data);
//        float[] CArray = new float[x * z];
//
//        mult(A, B, CArray, x, y, z);
//
//        // Convert float array back to double
//        convertToDouble(CArray, c.data);
//    }
//
//    /**
//     * Converts a double array to a float array.
//     *
//     * @param doubleArray Source double array
//     * @return Converted float array
//     */
//    private float[] convertToFloat(double[] doubleArray) {
//        float[] floatArray = new float[doubleArray.length];
//        for (int i = 0; i < doubleArray.length; i++) {
//            floatArray[i] = (float) doubleArray[i];
//        }
//        return floatArray;
//    }
//
//    /**
//     * Converts a float array to a double array.
//     *
//     * @param floatArray  Source float array
//     * @param doubleArray Destination double array
//     */
//    private void convertToDouble(float[] floatArray, double[] doubleArray) {
//        for (int i = 0; i < floatArray.length; i++) {
//            doubleArray[i] = floatArray[i];
//        }
//    }
//
//    /**
//     * Multiplies two matrices using float arrays.
//     *
//     * @param A Input matrix A as float array
//     * @param B Input matrix B as float array
//     * @param C Output matrix C as float array (A * B)
//     * @param x Number of rows in A and C
//     * @param y Number of columns in A and rows in B
//     * @param z Number of columns in B and C
//     */
//    public void mult(float[] A, float[] B, float[] C, int x, int y, int z) {
//        int TILE_SIZE = 16;
//
//        // Calculate global work sizes
//        int globalX = roundUp(x, TILE_SIZE);
//        int globalY = roundUp(z, TILE_SIZE);
//
//        int requiredSizeA = x * y;
//        int requiredSizeB = y * z;
//        int requiredSizeC = globalX * globalY; // Adjusted to match global work size
//
//        // Acquire buffers from the pool
//        CLBuffer<FloatBuffer> bufferA = acquireBuffer(bufferPoolA, requiredSizeA, CLMemory.Mem.READ_ONLY, CLMemory.Mem.ALLOCATE_BUFFER);
//        CLBuffer<FloatBuffer> bufferB = acquireBuffer(bufferPoolB, requiredSizeB, CLMemory.Mem.READ_ONLY, CLMemory.Mem.ALLOCATE_BUFFER);
//        CLBuffer<FloatBuffer> bufferC = acquireBuffer(bufferPoolC, requiredSizeC, CLMemory.Mem.WRITE_ONLY, CLMemory.Mem.ALLOCATE_BUFFER);
//
//        try {
//            // Write data to buffers
//            bufferA.getBuffer().put(A).rewind();
//            bufferB.getBuffer().put(B).rewind();
//
//            // Set kernel arguments
//            mm.setArg(0, x); // M
//            mm.setArg(1, z); // N
//            mm.setArg(2, y); // K
//            mm.setArg(3, bufferA);
//            mm.setArg(4, bufferB);
//            mm.setArg(5, bufferC);
//
//            // Define local work sizes
//            int localX = TILE_SIZE;
//            int localY = TILE_SIZE;
//
//            // Enqueue write buffers and kernel execution
//            queue.putWriteBuffer(bufferA, false)
//                    .putWriteBuffer(bufferB, false)
//                    .put2DRangeKernel(mm, 0, 0, globalX, globalY, localX, localY)
//                    .putReadBuffer(bufferC, true)
//                    .finish();
//
//            // Reset buffer position before reading
//            bufferC.getBuffer().rewind();
//
//            // Read back the result (only the actual matrix size, not the padded area)
//            for (int i = 0; i < x * z; i++) {
//                C[i] = bufferC.getBuffer().get(i);
//            }
//        } catch (CLException e) {
//            logger.error("OpenCL error during matrix multiplication", e);
//            throw e;
//        } finally {
//            // Release buffers back to the pool
//            releaseBuffer(bufferPoolA, bufferA);
//            releaseBuffer(bufferPoolB, bufferB);
//            releaseBuffer(bufferPoolC, bufferC);
//        }
//    }
//
//    /**
//     * Rounds up the global size to be a multiple of the local size.
//     *
//     * @param global Global size
//     * @param local  Local size
//     * @return Rounded up global size
//     */
//    private int roundUp(int global, int local) {
//        return (global + local - 1) / local * local;
//    }
//
//    /**
//     * Acquires a buffer from the pool or creates a new one if necessary.
//     *
//     * @param pool  Buffer pool
//     * @param size  Required size
//     * @param flags Memory flags
//     * @return CLBuffer instance
//     */
//    private CLBuffer<FloatBuffer> acquireBuffer(ConcurrentLinkedQueue<CLBuffer<FloatBuffer>> pool, int size, CLMemory.Mem... flags) {
//        CLBuffer<FloatBuffer> buffer = pool.poll();
//        if (buffer == null || buffer.getBuffer().capacity() < size) {
//            if (buffer != null) buffer.release();
//            buffer = cl.createFloatBuffer(size, flags);
//        }
//        return buffer;
//    }
//
//    /**
//     * Releases a buffer back to the pool.
//     *
//     * @param pool   Buffer pool
//     * @param buffer CLBuffer to release
//     */
//    private void releaseBuffer(ConcurrentLinkedQueue<CLBuffer<FloatBuffer>> pool, CLBuffer<FloatBuffer> buffer) {
//        pool.offer(buffer);
//    }
//
//    /**
//     * Main method for testing matrix multiplication.
//     *
//     * @param args Command-line arguments
//     */
//    public static void main(String[] args) {
//
//        try (var multiplier = new MatUtil_OpenCL()) {
//            for (int d : new int[]{1, 1, 1, /* warm up */ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048}) {
//                int x = d, y = d, z = d;
//                Util.time("mult_" + d, logger, () -> {
//                    multiplier.mult(
//                            new float[x * y],
//                            new float[y * z],
//                            new float[x * z],
//                            x, y, z);
//                });
//            }
//        }
//
//    }
//
//    /**
//     * Closes the OpenCL resources.
//     */
//    @Override
//    public void close() {
//        try {
//            if (mm != null && !mm.isReleased()) {
//                mm.release();
//            }
//            if (queue != null && !queue.isReleased()) {
//                queue.release();
//            }
//            if (cl != null && !cl.isReleased()) {
//                cl.release();
//            }
//        } catch (CLException e) {
//            logger.warn("Error releasing OpenCL resources", e);
//        }
//    }
//}
