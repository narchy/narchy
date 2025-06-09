package fom.util;


/*
 * Balanced Mutual Information (BMI)
 * calcola il grado di associazione tra due termini
 */
public class BMI implements CorrelationFunction{
	
	private final double beta;
	private static final double def_beta=0.51f;
	private static final PMI mi = new PMI();
	
	public BMI(){
		this(def_beta);
	}
	
	public BMI(double b){
		this.beta=b;
	}
	
	@Override
    public double calculateCorrelation(double p_x, double p_y, double p_x_y) {
		double p_nx_ny	= 1 - (p_x + p_y - p_x_y); 	
		double p_x_ny	= p_x - p_x_y;     			
		double p_nx_y	= p_y - p_x_y;				
		double p_nx		= 1 - p_x;					
		double p_ny		= 1 - p_y;


        double mi_x_y = mi.calculateCorrelation(p_x, p_y, p_x_y);

        double mi_nx_ny = mi.calculateCorrelation(p_nx, p_ny, p_nx_ny);

        double mi_x_ny = mi.calculateCorrelation(p_x, p_ny, p_x_ny);
        double mi_nx_y = mi.calculateCorrelation(p_nx, p_y, p_nx_y);

        double bmi = beta * (jcog.Util.fma(p_x_y, mi_x_y, p_nx_ny * mi_nx_ny))
                - (1 - beta) * (jcog.Util.fma(p_x_ny, mi_x_ny, p_nx_y * mi_nx_y));

		
		return bmi;
	}

}