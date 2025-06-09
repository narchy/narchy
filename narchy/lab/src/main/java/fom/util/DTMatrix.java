package fom.util;

import jcog.data.map.ObjIntHashMap;

import java.util.ArrayList;
import java.util.List;

/**
 * Classe che modella una matrice triangolare espandibile dinamicamente
 * 
 */
public class DTMatrix<E> {
	
	
	
	/**
	 * Costruttore base
	 */
	public DTMatrix(){
		this(minSize);
	}
	
	
	/**
	 * Costruttore con parametro
	 * @param size grandezza della matrice, che sarà size * (size + 1)/2
	 */
	public DTMatrix(int size){
		terms = new ObjIntHashMap<>(size);
		values = allocateMatrix(size);
	}
	
	/**
	 * Alloca una matrice triangolare inferiore
	 * @param size Grandezza della matrice
	 * @return Una matrice triangolare inferiore
	 */
	private double[][] allocateMatrix(int size)
	{
		List<double[]> list = new ArrayList<>(size);
		for (int i = 0; i < size; i++) {
			list.add(new double[i + 1]);
		}
		return list.toArray(new double[0][]);
	}
	
	
	/**
	 * Aumenta le dimensioni di una matrice
	 * @param matrix La matrice iniziale
	 * @return Una matrice grande il doppio, con gli stessi dati dell'originale
	 */
	private double[][] growSize(double[][] matrix)
	{
		int newSize = matrix.length * 2;
		double[][] newMatrix = allocateMatrix(newSize);
		for(int i = 0; i < matrix.length; i++){
			System.arraycopy(matrix[i], 0, newMatrix[i], 0, matrix[i].length);
		}
		return newMatrix;
	}
	
	/**
	 * Metodo che controlla se la grandezza della matrice dei valori è 
	 * abbastanza grande, se non lo è ne aumenta la grandezza.
	 */
	private void checkSize(){
		if(terms.size() > values.length)
			values = growSize(values);
	}
	
	
	/**
	 * Metodo che restituisce il valore associato al termine indicato
	 * @param term Il termine di cui si vuole conoscere il valore associato
	 * @return Il valore associato al termine target
	 */
	protected double getValue(E term){
		int i = index(term);
		return values[i][i];
	}
	
	/**
	 * Metodo che restituisce il valore associato ai termini indicati
	 * @param first Primo termine
	 * @param second Secondo termine
	 * @return Il valore associato a first e second 
	 */
	protected double getValue(E first, E second){
        int i = index(first);
        int j = index(second);

		if (i < j) {
			int tmp = i;
			i = j;
			j = tmp;
		}

		return values[i][j];
	}
	
	/**
	 * Metodo che imposta un valore per un dato termine
	 * @param term il valore di cui si vuole associare il valore
	 * @param value il valore da associare a target
	 */
	protected void setValue(E term, double value){
		int i = index(term);
		values[i][i] = value;
	}

	public int index(E term) {
		int i = terms.getIfAbsent(term, -1);
		if(i ==-1){
			i = terms.size();
			terms.put(term, i);
			checkSize();
		}
		return i;
	}

	/**
	 * Metodo che associa un valore ad una coppia di termini
	 * @param first il primo termine della coppia
	 * @param second il secondo termine della coppia
	 * @param value il valore da associare alla coppia
	 */
	protected void setValue(E first, E second, double value){
		int i = index(first);
		int j = index(second);

		if (i < j) {
			int tmp = i;
			i = j;
			j = tmp;
		}

		values[i][j] = value;
	}
	
	/**
	 * Metodo che incrementa il valore associato ad un termine
	 * @param term termine il cui valore va incrementato
	 * @param increment incremento
	 */
	protected void add(E term, double increment){
		int i = index(term);
		values[i][i] += increment;
	}
	
	/**
	 * Metodo che incrementa il valore associato ad una coppia di termini
	 * @param first primo termine
	 * @param second secondo termine
	 * @param increment incremento
	 */
	protected void add(E first, E second, double increment){
		int i = index(first);
		int j = index(second);
		if (i < j) {
			int tmp = i;
			i = j;
			j = tmp;
		}
		values[i][j] += increment;
	}
	
	protected void normalizeBy(double value){
        for(int i = 0; i < values.length; i++)
			for(int j = 0; j < values[i].length; j++){
				values[i][j] /= value;
			}
	}

    protected double[][] values;
	protected ObjIntHashMap<E> terms;
	private static final int minSize = 128;
}