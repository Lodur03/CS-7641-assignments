package Assigment02;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.*;
import util.linalg.Vector;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.text.DecimalFormat;
import java.util.*;
public class Spambase {
/*preprocess the data follow the convention*/
	
// Credits: https://github.com/XiaohuaCao/Randomized-Optimization
	
// TDDO: The dataset filename. Expected format is CSV
private static final String FILENAME = "/media/wd/Data/Xiaohua/ABAGAILEclipse/Spambase.csv";
	
//TODO: How many examples you have
private static int num_examples = 4601;
//TODO: How many attributes you have. This is (number_of_columns -1)
private static int num_attributes = 57;
/*Randomization
 * If you enable the randomization, your rows will be shuffled.
 * The seed value can be any arbitrary value. Having a seed ensures that each run of the script has the same randomized shuffle ordering as any other run.
 */
private static final boolean shouldRandomize = true;
private static final long SEED = 0xABCDEF;
/*Cross Validation and Testing params
 * Set K for number of folds.
 * Set PERCENT_TRAIN for the percentage to be use for training
 */

private static final double PERCENT_TRAIN = 0.7;
/* Neural Network Params
 * Tweak the following as needed. The maxLearningRate and minLearningRate are the default
 * The accuracy threshold to stop backprop. Set this EXTREMELY low to ensure training ends only
 * when absolutely sure network output is correct
 */
private static double initialLearningRate = 0.1;
private static double maxLearningRate = 50;
private static double minLearningRate = 0.000001;
private static double backprop_threshold = 1e-10;
/*Backprop Prams
 * If true, shouldFindNNParams determines the best params for your neural network. Since this process is lengthy, you don't want to repreat it often. Do it once, record the value, and place the value for numberHiddenLayersNodes.
 * 
 */
private static final boolean shouldFindNNParams = false;
private static int numberHiddenLayerNodes=20;
/*Simulated Annealing
 * Same as above, if you have already run it once, just record the values and store then in best_temp and best_cooling. Otherwise, the temps[] and cooling[] values will be cycled through until the best param configuratoin for simulated annealing is found.
 * 
 */
// TODO: set this to false if you retained the best SA params from a previous run
private static final boolean shouldFindSAParams = false;
//TODO: Modify these to try different possible temp and cooling params
private static double[] temps = {1e5, 1e8, 1e11, 1e13, 1e15};
private static double[] coolingRates = {0.9, 0.95, 0.99, 0.999};
//TODO: Place values here from previous run if you set shouldFindSAParams to false.
private static double best_temp = 1e11;
private static double best_cooling = 0.9;

/* Genetic Algorithms 
 * Same as above, if you have already run it once, just record the values and store them in 
 * populationSize, toMate, and toMutate. Otherwise, the ratios will be cycled through until the best param configuration for genetic algorithms is found.
 *NOTE: min(populationRations)>=max(mateRatios)+max(toMutateRatio)
 *This condition must be upheld or Exception will be thrown later in the script
 *
 */
private static final boolean shouldFindGAParams = false;
private static double[] populationRatios = {0.10, 0.15, 0.20, 0.25};
private static double[] mateRatios = {0.01,0.03,0.05,0.07,0.09,0.1};
private static double[] mutateRatios = {0.01,0.02,0.03};
// TODO: Place values here from previous run if you set shouldFindGAPrams to false
private static double populationRatio = 0.1;
private static double toMateRatio = 0.03;
private static double toMutateRatio = 0.1;

/* Other global vars - Don't mess with these unless you have a reason to 
 * Number of input nodes is the same as the number of attributes
 */
private static int inputLayer = num_attributes;
// This is determined dynamically
private static int outputLayer;
//Determines how many possible classifications we can have
private static Set<String> unique_labels;
private static Instance[] allInstances;
private static Map<String, double[]> bitvector_mappings;
private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
private static ErrorMeasure measure = new SumOfSquaresError();
private static DecimalFormat df = new DecimalFormat ("0.000");
// Train and Test sets
private static DataSet trainSet;
private static DataSet testSet;
// Cross validation folds

/* Begin actual code
 * Comment out anything you don't want to immediately run.
 * Write loops to do certain numbers of iterations and output results for rand opt
 * algorithms
 * @param args ignored
 */
public static void main(String[] args){
	initializeInstances();
// Create k-folds for cross-validation
	makeTestTrainSets();
	
	int trainIterations;
	
// Determine best NN params, which happen to just the number of hidden layer nodes.
	
	/*
	if(shouldFindNNParams){
		// TODO: Tweak this value for better results
		// This is how Weka does it: # of attributes + # of classes /2
		int weka_technique = (num_attributes+outputLayer)/2;
		int[] hiddenLayerNodes = {weka_technique, 5, 10, 15, 20, 35, 40};
		trainIterations = 500;
		determinNNParams(trainIterations, hiddenLayerNodes);
		
	}
	/* Backprop*/
	int[] numIterations = {1700, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500,3700, 3900, 4100, 4300, 4500, 4700, 4900, 5100, 5300, 5500, 5700, 5900};
	/*for (Integer number : numIterations)
	{
		// Now determine the NN performance. Results are simply printed out.
		runBackprop(number);
	}
	
	/*RHC*/
	
	/*for (Integer number : numIterations)
	{
		// RHC has no params, just run it directly
		runRHC(number);
	}
	
	/*SA*/
	if (shouldFindSAParams){
		trainIterations = 500;
		determineSAParams(trainIterations);
			
	}
	// Run actual SA with best params here
	
	/*for (Integer number : numIterations)
	{
		runSA(number, best_temp, best_cooling);
		
	}

	
	/* GA */
	if (shouldFindGAParams){
		// TODO: keep this very small
		trainIterations = 500;
		determineGAParams(trainIterations);
	}
	
	

	// Run actual GA with best params here
	
	for (Integer number : numIterations)
	{
		runGA(number,populationRatio,toMateRatio,toMutateRatio);
		
	}

	
	
	/**
	 * Uses k-folds cross valiation to determine the best number of hidden layers to configure a neural network among the list provide
	 *@param trainIterations the number of iterations to run
	@param hiddenLayerNodes an int[] of hidden layer nodes to try 
	 */
	
}

public static void determinNNParams(int trainIterations, int[] hiddenLayerNodes) {

    System.out.println("===========Test Validation for NN Params=========");
    double[] testErrors = new double[hiddenLayerNodes.length];
    double[] trainErrors = new double[hiddenLayerNodes.length];
    for (int m = 0; m < hiddenLayerNodes.length; m++) {

      int numHiddenNodes = hiddenLayerNodes[m];
      
//    Creates a network with specified number of hidden layer nodes
      BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
          new int[]{inputLayer, numHiddenNodes, outputLayer});
      ConvergenceTrainer trainer = new ConvergenceTrainer(
              new BatchBackPropagationTrainer(trainSet, backpropNet, new SumOfSquaresError(),
                  new RPROPUpdateRule(initialLearningRate, maxLearningRate, minLearningRate)),
              backprop_threshold, trainIterations);
      trainer.train();

      trainErrors[m] = evaluateNetwork(backpropNet, trainSet.getInstances());
      testErrors[m] = evaluateNetwork(backpropNet, testSet.getInstances());
      
      System.out.printf("Nodes: %d\tTrian Error: %s%%%n\tTest Error%s%% %n", numHiddenNodes, df.format(trainErrors[m]),df.format(testErrors[m]));
    }

//    Find the index with the min test error
    int best_index = 0;
    double minError = Double.MAX_VALUE;
    for (int j = 0; j < testErrors.length; j++) {
      if (testErrors[j] < minError) {
        minError = testErrors[j];
        best_index = j;
      }
    }
    int bestNumNodes = hiddenLayerNodes[best_index];

    System.out.printf("%nBest Num Hidden Nodes: %d\tError: %s%%%n", bestNumNodes, df.format
        (minError));

    numberHiddenLayerNodes = bestNumNodes;
  }
/**
 * This method will run Backpropagation using each
 * combination of (K-1) folds for training, and the Kth fold for validation. Once the model
 * with the lowest validation set error is found, that is used as the "best" model and the
 * training and test errors on that model are recorded.
 * @param trainIterations the number of epochs/iterations
 */
public static void runBackprop(int trainIterations) {

    System.out.println("===========Backpropagation=========");
    System.out.println("NumberHiddenLayerNodes: "+numberHiddenLayerNodes );

    // training and calculate training error and training time
    double starttime_train = System.nanoTime();
    double endtime_train;

    BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
        new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

    ConvergenceTrainer trainer = new ConvergenceTrainer(
        new BatchBackPropagationTrainer(trainSet, backpropNet, new SumOfSquaresError(),
            new RPROPUpdateRule(initialLearningRate, maxLearningRate, minLearningRate)),
        backprop_threshold, trainIterations);

    trainer.train();

    double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
    endtime_train = System.nanoTime();
    
    double time_elapsed_train = endtime_train - starttime_train;
    
    // Calculate Test error and test time
    
    double starttime_test = System.nanoTime();
    double endtime_test;
    double testError = evaluateNetwork(backpropNet, testSet.getInstances());
    endtime_test = System.nanoTime();
    double time_elapsed_test = endtime_test - starttime_test;

//    Convert nanoseconds to seconds
    time_elapsed_train /= Math.pow(10,9);
    time_elapsed_test /= Math.pow(10,9);
    System.out.printf(trainIterations + " Time Elapsed_train: %s s %n", df.format(time_elapsed_train));
    System.out.printf(trainIterations + " Time Elapsed_test: %s s %n", df.format(time_elapsed_test));
    System.out.printf(trainIterations+" Train Error: %s%% %n", df.format(trainError));
    System.out.printf(trainIterations + " Test Error: %s%% %n", df.format(testError));  

  }

/**
 * Determines optimal weights for configured neural network using Randomized Hill Climbing with
 * Random Restarts and evaluates a neural networks performance on train and test sets with
 * those weights
 * @param trainIterations the number of iterations
 */

public static void runRHC(int trainIterations) {

    System.out.println("===========Randomized Hill Climbing=========");
    System.out.println("NumberHiddenLayerNodes: "+numberHiddenLayerNodes );
    double starttime_train = System.nanoTime();
    double endtime_train;

    BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
        new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

    NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
        backpropNet, measure);

    OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);

//      TODO: Vary the number of iterations as needed for your results
    train(oa, backpropNet, trainIterations);

    double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
    endtime_train = System.nanoTime();
    
    double time_elapsed_train = endtime_train - starttime_train;
    
    // Calculate Test error and test time
    
    double starttime_test = System.nanoTime();
    double endtime_test;
    double testError = evaluateNetwork(backpropNet, testSet.getInstances());



    endtime_test = System.nanoTime();
    double time_elapsed_test = endtime_test - starttime_test;

//    Convert nanoseconds to seconds
    time_elapsed_train /= Math.pow(10,9);
    time_elapsed_test /= Math.pow(10,9);
    System.out.printf(trainIterations + "Train Error: %s%% %n", df.format(trainError));
    System.out.printf(trainIterations + "Test Error: %s%% %n", df.format(testError));
    System.out.printf(trainIterations + "Time Elapsed_train: %s s %n", df.format(time_elapsed_train));
    System.out.printf(trainIterations + "Time Elapsed_test: %s s %n", df.format(time_elapsed_test));

  }


// Optimization of SA algoirhtm with varied coolingRates and temperatures
  public static void determineSAParams(int trainIterations) {

    System.out.println("===========Determining Simulated Annealing Params=========");
    System.out.println("NumberHiddenLayerNodes: "+numberHiddenLayerNodes );
   
    double[][] testErrors = new double[temps.length][coolingRates.length];
    double[][] trainErrors = new double[temps.length][coolingRates.length];

    for (int x = 0; x < temps.length; x++) {

      double temp = temps[x];

      for (int y = 0; y < coolingRates.length; y++) {

        double cooling = coolingRates[y];
        
//      Creates a network with specified number of hidden layer nodes       
        BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
                new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});
        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, backpropNet, measure);
        OptimizationAlgorithm oas = new SimulatedAnnealing(temp, cooling, nnop);
        train(oas, backpropNet, trainIterations);

        trainErrors[x][y] = evaluateNetwork(backpropNet, trainSet.getInstances());
        testErrors[x][y] = evaluateNetwork(backpropNet, testSet.getInstances());
        System.out.println("temp: %d" + temp);
        System.out.println("colling: "+cooling);
        System.out.printf("Trian Error: %s%%%n\tTest Error%s%% %n", df.format(trainErrors[x][y]),df.format(testErrors[x][y]));
        
      }
    }
    int best_temp_index = 0;
    int best_cool_index = 0;
    double minErr = Double.MAX_VALUE;

    for (int x = 0; x < temps.length; x++) {

      for (int y = 0; y < coolingRates.length; y++) {

        if (minErr > testErrors[x][y]) {
          best_temp_index = x;
          best_cool_index = y;
          minErr = testErrors[x][y];
        }
      }
    }

    double bestCooling = coolingRates[best_cool_index];
    double bestTemp = temps[best_temp_index];
    System.out.printf("Best Cooling: %s%n", df.format(bestCooling));
    System.out.printf("Best Temp: %s%n", df.format(bestTemp));

  }
  
  // Run SA
  
  public static void runSA(int trainIterations, double temp, double cooling) {

	    System.out.println("===========Simulated Annealing=========");
	    System.out.println("NumberHiddenLayerNodes: "+numberHiddenLayerNodes );
	    

	    double starttime_train = System.nanoTime();;
	    double endtime_train;

	    BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
	        new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

	    NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
	        backpropNet, measure);

	    OptimizationAlgorithm oa = new SimulatedAnnealing(temp, cooling, nnop);

	    train(oa, backpropNet, trainIterations);
	    double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
	    
	    endtime_train = System.nanoTime();
	    
	    double time_elapsed_train = endtime_train - starttime_train;
	    
	    double starttime_test = System.nanoTime();;
	    double endtime_test;
	    
	    
	    double testError = evaluateNetwork(backpropNet, testSet.getInstances());


	    endtime_test = System.nanoTime();
	    double time_elapsed_test = endtime_test - starttime_test;

//	    Convert nanoseconds to seconds
	    time_elapsed_train /= Math.pow(10,9);
	    time_elapsed_test /= Math.pow(10,9);

	    System.out.printf(trainIterations + " Train Error: %s%% %n", df.format(trainError));
	    System.out.printf(trainIterations + " Test Error: %s%% %n", df.format(testError));
	    System.out.printf(trainIterations + " Time Elapsed: %s s %n", df.format(time_elapsed_train));
	    System.out.printf(trainIterations + " Time Elapsed: %s s %n", df.format(time_elapsed_test));
	  }
// Optimization GA algorithm
  
  public static void determineGAParams(int trainIterations) {

	    System.out.println("===========Determining Genetic Algorithms Params=========");
	    System.out.println("NumberHiddenLayerNodes: "+numberHiddenLayerNodes );
	    
	    double[][][] testErrors = new double[populationRatios.length][mateRatios.length][mutateRatios.length];
	    double[][][] trainErrors = new double[populationRatios.length][mateRatios.length][mutateRatios.length];
	    
//	    Training population size is always 9/10 of the total training set, or equivalently
//	    9 times the validation set


	    for (int x = 0; x < populationRatios.length; x++) {

	      int population = (int) (populationRatios[x] * trainSet.size());

	      for (int y = 0; y < mateRatios.length; y++) {

	        int mate = (int) (mateRatios[y] * trainSet.size());

	        for (int z = 0; z < mutateRatios.length; z++) {

	          int mutate = (int) (mutateRatios[z] * trainSet.size());
	          
//	          Creates a network with 3 different parameters  
	          
	          BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
	                  new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});
	          NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet, backpropNet, measure);
	          OptimizationAlgorithm oas = new StandardGeneticAlgorithm(population, mate, mutate, nnop);
	          train(oas, backpropNet, trainIterations);

	          trainErrors[x][y][z] = evaluateNetwork(backpropNet, trainSet.getInstances());
	          testErrors[x][y][z] = evaluateNetwork(backpropNet, testSet.getInstances());
	          
	          System.out.println("pupulationRatios: %f"+populationRatios[x]);
	          System.out.println("mateRatios: %f"+ mateRatios[y]);
	          System.out.println("mutateRatios: %f"+mutateRatios[z]);
	          System.out.printf("Train Error: %s%% %n", df.format(trainErrors[x][y][z]));
	          System.out.printf("Test Error: %s%% %n", df.format(testErrors[x][y][z]));
	        }
	      }
	    }
	          	         
	    int best_pop = 0;
	    int best_mate = 0;
	    int best_mutate = 0;
	    double minErr = Double.MAX_VALUE;

	    for (int x = 0; x < populationRatios.length; x++) {

	      for (int y = 0; y < mateRatios.length; y++) {

	        for (int z = 0; z < mutateRatios.length; z++) {

	          if (testErrors[x][y][z]<minErr){
	        	  best_pop = x;
	        	  best_mate = y;
	        	  best_mutate = z;
	        	  minErr = testErrors[x][y][z];
	          } 
	        }
	      }
	    }
	   

	    populationRatio = populationRatios[best_pop];
	    toMateRatio = mateRatios[best_mate];
	    toMutateRatio = mutateRatios[best_mutate];
	    

	    System.out.printf("best Population Ratio: %s%n", df.format(populationRatio));
	    System.out.printf("best Mate Ratio: %s%n", df.format(toMateRatio));
	    System.out.printf("best Mutate Ratio: %s%n", df.format(toMutateRatio));
	    System.out.printf("minErrorr: %s%% %n", df.format(minErr));

	  }
  /**
   * Run genetic algorithms.
   * @param trainIterations the iterations to run
   */
  public static void runGA(int trainIterations,double populationRatio, double toMateRatio, double toMutateRatio) {

	    System.out.println("===========Genetic Algorithms=========");
	    System.out.println("NumberHiddenLayerNodes: "+numberHiddenLayerNodes );

	    double starttime_train = System.nanoTime();
	    double endtime_train;

	    int trainSetSize = trainSet.size();
	    int populationSize = (int) (trainSetSize * populationRatio);
	    int toMate = (int) (trainSetSize * toMateRatio);
	    int toMutate = (int) (trainSetSize * toMutateRatio);

	    BackPropagationNetwork backpropNet = factory.createClassificationNetwork(
	        new int[]{inputLayer, numberHiddenLayerNodes, outputLayer});

	    NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(trainSet,
	        backpropNet, measure);

	    OptimizationAlgorithm oa = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, nnop);

	    train(oa, backpropNet, trainIterations);


	    double trainError = evaluateNetwork(backpropNet, trainSet.getInstances());
	    endtime_train = System.nanoTime();
	    double time_elapsed_train = endtime_train - starttime_train;
	    
	    double starttime_test = System.nanoTime();
	    double endtime_test;
	    double testError = evaluateNetwork(backpropNet, testSet.getInstances());

	    System.out.printf(trainIterations+ "Train Error: %s%% %n", df.format(trainError));
	    System.out.printf(trainIterations+"Test Error: %s%% %n", df.format(testError));

	    endtime_test = System.nanoTime();
	    double time_elapsed_test = endtime_test - starttime_test;

//	    Convert nanoseconds to seconds
	    time_elapsed_train /= Math.pow(10,9);
	    time_elapsed_test /= Math.pow(10,9);
	    System.out.printf(trainIterations+ "Train Error: %s%% %n", df.format(trainError));
	    System.out.printf(trainIterations+"Test Error: %s%% %n", df.format(testError));
	    
	    System.out.printf(trainIterations+"Time Elapsed_train: %s s %n", df.format(time_elapsed_train));
	    System.out.printf(trainIterations+"Time Elapsed_test: %s s %n", df.format(time_elapsed_test));
	  }
  /**
   * Train a given optimization problem for a given number of iterations. Called by RHC, SA, and
   * GA algorithms
   * @param oa the optimization algorithm
   * @param network the network that corresponds to the randomized optimization problem. The
   *                optimization algorithm will determine the best weights to try using with this
   *                network and assign those weights
   * @param iterations the number of training iterations
   */
  private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, int iterations)
  {
	  for(int i=0; i<iterations; i++){
		  oa.train();
	  }
	  Instance optimalWeights = oa.getOptimal();
	  network.setWeights(optimalWeights.getData());
  }
  /**
   * Given a network and instances, the output of the network is evaluated and a decimal value
   * for error is given
   * @param network the BackPropagationNetwork with weights already initialized
   * @param data the instances to be evaluated against
   * @return
   */
  public static double evaluateNetwork(BackPropagationNetwork network, Instance[] data) {

	    double num_incorrect = 0;

	    for (int j = 0; j < data.length; j++) {
	      network.setInputValues(data[j].getData());
	      network.run();

	      Vector actual = data[j].getLabel().getData();
	      Vector predicted = network.getOutputValues();


	      boolean mismatch = ! isEqualOutputs(actual, predicted);

	      if (mismatch) {
	        num_incorrect += 1;
	      }
	    }

	    double error = num_incorrect / data.length * 100;
	    return error;

	  }
  /**
   * Compares two bit vectors to see if expected bit vector is most likely to be the same
   * class as the actual bit vector
   * @param actual
   * @param predicted
   * @return
   */
  private static boolean isEqualOutputs(Vector actual, Vector predicted) {

	    int max_at = 0;
	    double max = 0;

//	    Where the actual max should be
	    int actual_index = 0;

	    for (int i = 0; i < actual.size(); i++) {
	      double aVal = actual.get(i);

	      if (aVal == 1.0) {
	        actual_index = i;
	      }

	      double bVal = predicted.get(i);

	      if (bVal > max) {
	        max = bVal;
	        max_at = i;
	      }
	    }

	    return actual_index == max_at;

	  }
  /**
   * Reads a file formatted as CSV. Takes the labels and adds them to the set of labels (which
   * later helps determine the length of bit vectors). Records real-valued attributes. Turns the
   * attributes and labels into bit-vectors. Initializes a DataSet object with these instances.
   */
  private static void initializeInstances() {

	    double[][] attributes = new double[num_examples][];

	    String[] labels = new String[num_examples];
	    unique_labels = new HashSet<>();


//	    Reading dataset
	    try {
	      BufferedReader br = new BufferedReader(new FileReader(new File(FILENAME)));

//	      You don't need these headers, they're just the column labels

	      String useless_headers = br.readLine();

	      for(int i = 0; i < attributes.length; i++) {
	        Scanner scan = new Scanner(br.readLine());
	        scan.useDelimiter(",");

	        attributes[i] = new double[num_attributes];

	        for(int j = 0; j < num_attributes; j++) {
	          attributes[i][j] = Double.parseDouble(scan.next());
	        }

//	        This last element is actually your classification, which is assumed to be a string
	        labels[i] = scan.next();
	        unique_labels.add(labels[i]);
	      }
	    }
	    catch(Exception e) {
	      e.printStackTrace();
	    }


//	    Creating a mapping of bitvectors. So "some classification" => [0, 1, 0, 0]
	    int distinct_labels = unique_labels.size();
	    outputLayer = distinct_labels;

	    bitvector_mappings = new HashMap<>();

	    int index = 0;
	    for (String label : unique_labels) {
	      double[] bitvect = new double[distinct_labels];

//	      At index, set to 1 for a given string
	      bitvect[index] = 1.0;
//	      Increment which index will have a bit flipped in next classification
	      index++;

	      bitvector_mappings.put(label, bitvect);
	    }

//	    Replaces the label for each instance with the corresponding bit vector for that label
//	    This works even for binary classification
	    allInstances = new Instance[num_examples];
	    for (int i = 0; i < attributes.length; i++) {
	      double[] X = attributes[i];

	      String label = labels[i];
	      double[] bitvect = bitvector_mappings.get(label);

	      Instance instance = new Instance(X);
	      instance.setLabel(new Instance(bitvect));

	      allInstances[i] = instance;
	    }
	  }


	  /**
	   * Print out the actual vs expected bit-vector. Used for debugging purposes only
	   * @param actual what the example's actual bit vector looks like
	   * @param expected what a network output as a bit vector
	   */
	  public static void printVectors(Vector actual, Vector expected) {
	    System.out.print("Actual: [");
	    for (int i = 0; i < actual.size(); i++) {
	      System.out.printf(" %f", actual.get(i));
	    }
	    System.out.print(" ] \t Expected: [");

	    for (int i = 0; i < expected.size(); i++) {
	      System.out.printf(" %f", expected.get(i));
	    }
	    System.out.println(" ]");
	  }


	  /**
	   * Takes all instances, and randomly orders them. Then, the first PERCENT_TRAIN percentage of
	   * instances form the trainSet DataSet, and the remaining (1 - PERCENT_TRAIN) percentage of
	   * instances form the testSet DataSet.
	   */
	  public static void makeTestTrainSets() {

	    List<Instance> instances = new ArrayList<>();

	    for (Instance instance: allInstances) {
	      instances.add(instance);
	    }

	    Random rand = new Random(SEED);
	    if (shouldRandomize) {
	      Collections.shuffle(instances, rand);
	    }

	    int cutoff = (int) (instances.size() * PERCENT_TRAIN);

	    List<Instance> trainInstances = instances.subList(0, cutoff);
	    List<Instance> testInstances = instances.subList(cutoff, instances.size());

	    Instance[] arr_trn = new Instance[trainInstances.size()];
	    trainSet = new DataSet(trainInstances.toArray(arr_trn));

	    System.out.println("Train Set: "+trainSet.size());

	    Instance[] arr_tst = new Instance[testInstances.size()];
	    testSet = new DataSet(testInstances.toArray(arr_tst));
	    System.out.println("Test Set: "+ testSet.size());

	  }


	  /**
	   * Given a DataSet of training data, separate the instances into K nearly-equal-sized
	   * partitions called folds for K-folds cross validation
	   * @param training, the training DataSet
	   * @return a list of folds, where each fold is an Instance[]
	   */
	 /** public static List<Instance[]> kfolds(DataSet training) {

	    Instance[] trainInstances = training.getInstances();

	    List<Instance> instances = new ArrayList<>();
	    for (Instance instance: trainInstances) {
	      instances.add(instance);
	    }

	    List<Instance[]> folds = new ArrayList<>();

//	    Number of values per fold
	    int per_fold = (int) Math.floor((double)(instances.size()) / K);

	    int start = 0;
	    int end = per_fold;

	    while (start < instances.size()) {


	      List<Instance> foldList = null;

	      if (end > instances.size()) {
	        end = instances.size();
	      }
	      foldList = instances.subList(start, end);

	      Instance[] fold = new Instance[foldList.size()];
	      fold = foldList.toArray(fold);

	      folds.add(fold);

	      start = end + 1;
	      end = start + per_fold;

	    }
	    System.out.println("the Folds: "+folds.size());
	    return folds;
	    
	  }
	  **/


	  /**
	   * Given a list of Instance[], this helper combines each arrays contents into one, single
	   * output array
	   * @param instanceList the list of Instance[]
	   * @return the combined array consisting of the contents of each Instance[] in instanceList
	   */
	  public static Instance[] combineInstances(List<Instance[]> instanceList) {
	    List<Instance> combined = new ArrayList<>();

	    for (Instance[] fold: instanceList) {

	      for (Instance instance : fold) {
	        combined.add(instance);
	      }
	    }

	    Instance[] combinedArr = new Instance[combined.size()];
	    combinedArr = combined.toArray(combinedArr);
	    return combinedArr;
	  }


	  /**
	   * Given a list of folds and an index, it will provide an Instance[] with the combined
	   * instances from every fold except for the fold at the given index
	   * @param folds the K-folds, a list of Instance[] used as folds for cross-validation
	   * @param foldIndex the index of the fold to exclude. That fold is used as the validation set
	   * @return the training folds combined into once Instance[]
	   */
	  public static Instance[] getTrainFolds(List<Instance[]> folds, int foldIndex) {
	    List<Instance[]> trainFolds = new ArrayList<>(folds);
	    trainFolds.remove(foldIndex);

	    Instance[] trnfolds = combineInstances(trainFolds);
	    return trnfolds;
	  }


	  /**
	   * Given a list of folds and an index, it will provide an Instance[] to serve as a validation
	   * set.
	   * @param folds the K-folds, a list of Instance[] used as folds for cross-validation
	   * @param foldIndex the index of the fold to use as the validation set
	   * @return the validation set
	   */
	  public static Instance[] getValidationFold(List<Instance[]> folds, int foldIndex) {
	    return folds.get(foldIndex);
	  }


	}
  










