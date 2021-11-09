include(pwd()*raw"\GRASPEL.jl")

vars = matread(pwd()*raw"\USPS.mat") #read in matfile with USPS data
fea = vars["fea"] #read in matfile matrix as a Vector
gnd = vec(vars["gnd"].-1) #Read in matfile gnd, labels in gnd are offset by 1?? Also convert to vector
#Graspel = 1 #do graspel iters? unused currently
kNN = 2 #for setting up initial kNNT
samplePerc = 1/100 #percentage of sorted nodes as candidate edges
edge_iter = .01/100 #edge budget per iteration (percentage of nodes)
sig = 1e3 #feature variance
embedTol = 10 #embedding distortion tolerance (should be > 1)
maxIterations = 100 #max number of GRASPEL maxIterations

Subset = [0:1:9;]
num_cluster = length(Subset)
num_eigs = 5*num_cluster
num_test = 10 #number of kmean clustering runs
num_res_test = 1000 #number of resistance TestResistance

#order dataset into subsets
num_node, dim = size(fea)
p = sortperm(gnd)
fea = fea[p,:]
#feaVec = [vec(fea[i,:]) for i = 1:num_node]#fea must be a vector of vectors 
gnd = gnd[p]
#preprocessing
##center images by average value
for i = 1:num_node
    fea[i,:] = fea[i,:].-mean(fea[i,:])
end
##normalize
##Julia norm(A,2) returns Frobenius Norm... not the same as Matlab's default 2-norm
##max(svd(A)), in this case svdvals(A)[1] , should eventually test this with larger data sets
fea = (fea/svdvals(fea)[1])*sqrt(dim)

#Do GRASPEL
G, varRatio, costVec, numIter = FiedlerConstruct(fea, samplePerc, edge_iter, sig, maxIterations, kNN, embedTol, num_node, dim)

num_edge=(length(edges(G)));
density=num_edge/num_node
L_J = laplacian_matrix(G) 
L_J = L_J + (1/(sig^2))*sparse(LinearAlgebra.I,L_J.m,L_J.n)
A_J = adjacency_matrix(G)
mmwrite(pwd()*raw"\Julia_L.mtx",L_J)
mmwrite(pwd()*raw"\Julia_A.mtx",A_J)
@info "Density: $density"

TestResistance = 1 #resistance for Reff, check against data Distances