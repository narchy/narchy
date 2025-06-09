package jcog.agent;

import jcog.signal.ITensor;
import jcog.signal.tensor.TensorRing;
import jcog.signal.tensor.TensorSerial;

import java.util.List;

public class SensorTensor {

	protected final ITensor sensors;

	/** TODO array */
	protected final List<ITensor> _sensors;

	protected final TensorSerial sensorsNow;

	public int volume() {
		return sensors.volume();
	}

	public SensorTensor(List<ITensor> _sensors, int history) {
		this._sensors = _sensors;
		sensorsNow = new TensorSerial(_sensors);
		this.sensors = history > 0 ? new TensorRing(sensorsNow.volume(), history) : sensorsNow;
	}

	public ITensor update() {
		/* @Deprecated  */
		for (ITensor _sensor : _sensors) _sensor.snapshot();

		sensorsNow.update();

		if (sensors instanceof TensorRing tr)
			tr.setSpin(sensorsNow.snapshot()); //HACK bad

		return sensors;
	}

}