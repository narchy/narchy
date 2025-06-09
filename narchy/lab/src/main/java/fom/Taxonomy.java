package fom;

import java.util.ArrayList;
import java.util.Iterator;

public class Taxonomy<E> implements Iterable<E>{

	private final double[][] relations;
    private final ArrayList<E> concepts;
   




    
    public Taxonomy(ArrayList<E> concepts){
    	this.concepts = concepts;
    	int size = concepts.size();
    	relations = new double[size][size];
    }
    
    
    public void setSpecificity(E subConcept, E concept, double specificity){
        int i = concepts.indexOf(subConcept);
        int j = concepts.indexOf(concept);
    	relations[i][j] = specificity;
    }
    
    public double getSpecificity(E subConcept, E concept){
        int i = concepts.indexOf(subConcept);
        int j = concepts.indexOf(concept);
    	return relations[i][j];
    }

    public void taxonomyPruning(){
        int i,j;
        int numConcepts = concepts.size();

        /* conservo solo le relazioni più forti:
    	 * ovvero la relazione più alta fra rel(i,j) e rel(j,i)
    	 */
    	for(i = 0; i < numConcepts; i++){
    		for(j = i; j < numConcepts; j++){
    			if( relations[i][j] != relations[j][i])
	    			if( relations[i][j] > relations[j][i])
	    				relations[j][i] = 0;
	    			else
	    				relations[i][j] = 0;
    		}
    	}


        double[][] distance = new double[numConcepts][numConcepts];
        int bigNum = 10000;
        for(i = 0; i < numConcepts; i++)
    		for(j = 0; j < numConcepts; j++){
    			
    			if(i == j){
    				distance[i][j] = 0;
    				continue;
    			}
    			
    			distance[i][j] = (relations[i][j] > 0) ? relations[i][j]:bigNum; 
    		}
    	
    	/*
    	 * Ciò che segue è una leggera modifica dell'algoritmo di 
    	 * Floyd Warshall per all-pairs shortest path.
    	 */
    	for(int k = 0; k < numConcepts; k++){
    		for(i = 0; i < numConcepts; i++){
    			for (j = 0; j < numConcepts; j++) {
    				
    				/*
    				 * Questo controllo viene aggiunto per fare in modo che non 
    				 * vengano considerati gli archi diretti fra i e j  e neanche
    				 * gli archi il cui peso è minore dell'arco diretto quando 
    				 * si cerca lo shortest path che li collega: così facendo 
    				 * saremo in grado di sapere se esiste un path che non
    				 * comprende archi il cui peso è minore di quello diretto
    				 * e quindi decidere se esso va eliminato o meno.
    				 */
					if(relations[i][k] > relations[i][j] && relations[k][j] > relations[i][j]){
                        double sumDist = distance[i][k] + distance[k][j];
                        if(  sumDist < distance[i][j])
							distance[i][j] = sumDist;
					}
				}
    		}
    	}
    	
    	for(i = 0; i < numConcepts; i++)
    		for(j = 0; j < numConcepts; j++){
    			if((distance[i][j] > relations[i][j]) && relations[i][j] > 0){
    				System.out.println(concepts.get(i) + " - " + concepts.get(j) + " distance: " + distance[i][j] + " relation: " + relations[i][j]);
    				relations[i][j] = 0;			
    			}
    		}
    	
    	
    	/*for(i = 0; i < numConcepts; i++)
    		for(j = 0; j < numConcepts; j++)
    			if((distance[i][j] != relations[i][j]) && relations[i][j] > 0)
    				System.out.println(concepts.get(i) + " - " + concepts.get(j) + " distance: " + distance[i][j] + " relation: " + relations[i][j]);
    		*/	
    }
    
	@Override
	public Iterator<E> iterator() {
		return concepts.iterator();
	}
	
	public ArrayList<E> getConcepts(){
		return concepts;
	}

}