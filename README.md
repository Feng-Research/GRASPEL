# GraspelJulia
Implementation of GRASPEL algorithm into Julia
Preprocessing and loading set up for included USPS dataset as matfile (USPS.mat)

include GRASPEL.JL for the primary function which returns
G, a, varRatio[1:numIter], costVec[1:numIter], numIter

FiedlerConstruct(fea, samplePerc, edge_iter, sig, maxIterations, k, tol, num_node, dim)

	fea is the vector of n-dimensional inputs
	samplePerc is the percentage of sorted nodes as candidate edges
	edge_iter is the edge budget per iteration (percentage of nodes)
	sig is the feature variance
	maxIterations is the max number of GRASPEL iterations to perform
	k is the positive integer size of the original k-nearest-neighbor graph
	tol is the embedding distortion tolerance (should be > 1)
	num_node is the length of the input vector
	dim is the length of an element of the input vector

This function returns G, varRatio[1:numIter], costVec[1:numIter], numIter

	G is a SimpleWeightedGraphs object of the resulting graph
	varRatio is a history of the varRatio through the iterations
	costVec is a history of the costVec through the iterations
	numIter is the number of completed iterations

------------USPS Example
The example RUN_USPS.jl preprocesses and runs the data in the 
USPS.mat file, which is the USPS dataset of handwritten digits.

Lines 7-12 in the RUN_USPS dataset set the parameters in the FiedlerConstruct function

-----------Running the example
1) GRASPEL.jl, RUN_USPS.jl, and USPS.mat must be in the same directory

2) The following packages must be installed in the Julia REPL

	--MAT https://github.com/JuliaIO/MAT.jl  
	--HNSW https://github.com/JuliaNeighbors/HNSW.jl  
	--Distances https://github.com/JuliaStats/Distances.jl  
	--MatrixMarket https://github.com/JuliaSparse/MatrixMarket.jl  
	--Graphs https://github.com/JuliaGraphs/Graphs.jl  
	--SimpleWeightedGraphs https://github.com/JuliaGraphs/SimpleWeightedGraphs.jl  
	--Arpack https://github.com/JuliaLinearAlgebra/Arpack.jl  
	--Clustering https://github.com/JuliaStats/Clustering.jl  

	command:

		add MAT, HNSW, Distances, MatrixMarket, Graphs, SimpleWeightedGraphs, Arpack, Clustering 

	after opening the REPL and pressing "]" to enter pkg editing  

3) Navigate to the directory in terminal and run RUN_USPS.jl using the command:
"julia RUN_USPS.jl"

4) The function should create two matrixmarket (.mtx) files named "Julia_A.mtx" and "Julia_L.mtx", for the 
Laplacian and Adjacency matrices of the resulting graph G.