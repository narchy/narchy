package java4k.pinball4k.editor;

public class ObjectProperties {
	
	/** Is object visible or not */
	public boolean visible = true;

	/** Is object collidable or not */
	public boolean collidable = true;
	
	/** The score to add object is triggered */
	public int score;
	
	/** The bounce factor  */
	public float bounce = 0.75f;	
	
	/** Object behaviour */
	public int behaviorId;
}
