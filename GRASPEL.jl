using LinearAlgebra, MAT, SparseArrays
using Statistics, HNSW, Distances, MatrixMarket
using Graphs, SimpleWeightedGraphs
using Arpack, Clustering

function undirectedEdges(idxs,dists) #O(nk)
    num_node = length(idxs) #each node 
    k = length(idxs[1]) #nearest neighbors
    edges_dict = Dict{Tuple{Int,Int},Float64}()
    #edges = Vector{Tuple{Int,Int,Float64}}(undef,num_node*(k-1))
    idxs = convert(Array{Array{Int}},idxs) #idxs to integers
    for i = 1:num_node
        for j = 2:k
            if(dists[i][1]!=0.0)
                @warn "Node $i could not find itself: ef_construcion, M or ef in HNSW construction may be too low."
                a = min(i,idxs[i][j-1])
                b = max(i,idxs[i][j-1])
                edges_dict[(a,b)] = dists[i][j-1]
            else
                a = min(idxs[i][1],idxs[i][j])
                b = max(idxs[i][1],idxs[i][j])
                edges_dict[(a,b)] = dists[i][j]
            end
        end
    end
    sort(collect(edges_dict), by = x -> x[1])
end

function getLaplacian(G)
    A = adjacency_matrix(G)
    D = sparse(Diagonal(vec(sum(A;dims = 2))))
    L = D-A
    return L
end

function FiedlerAddEdges(G,fea,samplePerc,chosenPercEachIter,sig,embedTol)

    num_eigs_f = 5
    num_node,dim = size(fea)

    #=
    @info "laplacian_matrix"
    @time begin
    L = laplacian_matrix(G) 
    end
    =#
    #@info "laplacian_matrix"
    #@time begin
    L = getLaplacian(G)
    #end

    L = L + (1/(sig^2))*sparse(LinearAlgebra.I,L.m,L.n)
    #mmwrite(raw"C:\Users\jacec\Desktop\code\code\JuliaL.mtx",L)
    #L = mmread(raw"C:\Users\jacec\Documents\MATLAB\GRASPEL_Code\code\JuliaL.mtx")
    vals, vecs = eigs(L; nev = num_eigs_f, which = :SM, tol = 1e-6)
    #@info vals
    #@info vecs[1,:]
    for i in 2:num_eigs_f
        vecs[:,i] = vecs[:,i]/sqrt(vals[i])  
    end
    #vecs[:,2:num_eigs_f] = vecs[:,2:num_eigs_f].*[sqrt(i) for i in vals[2:num_eigs_f]]
    #@info vecs[1,:]

    #Fiedler Vector
    l2 = vecs[:,2]

    #Graph node sorting
    I = sortperm(l2)
    #@info I
    
    top = I[1:Int64(floor(num_node*samplePerc))]
    bot = I[num_node-Int64(floor(num_node*samplePerc)+1):num_node]
    chosenSet = zeros(Float64,Int64(ceil(num_node*chosenPercEachIter)),5)
    minDelta, location = findmin(chosenSet[:,3])

    #sample candidate edges

    for i in 1:length(chosenSet)*1000
        a_idx = rand(top);
        b_idx = rand(bot);
        #always set p to have the bigger node index
        p = max(a_idx, b_idx)
        q = min(a_idx, b_idx)
        #e_pq = zeros(num_node,1)       unsure what these were for
        #e_pq[p] = 1
        #e_pq[q] = -1
        distX = pairwise(SqEuclidean(),reshape(fea[p,:],:,1),reshape(fea[q,:],:,1))[1]
        weight = 1/distX
        distZ = pairwise(SqEuclidean(),reshape(vecs[p,:],:,1),reshape(vecs[q,:],:,1))[1]*dim
        delta_fiedler = distZ*weight
        #check if edge has been selected
        flagP = p in chosenSet[:,1]
        flagQ = q in chosenSet[:,2]
        if delta_fiedler > minDelta && !(flagP || flagQ) && delta_fiedler > embedTol
            chosenSet[location,1] = p
            chosenSet[location,2] = q
            chosenSet[location,3] = delta_fiedler
            chosenSet[location,4] = weight*1 #scale weight by up to 10
            gradient = (1-1/delta_fiedler)*distZ
            chosenSet[location,5] = gradient #gradient
            minDelta, location = findmin(chosenSet[:,3])
            #@info "P:$p Q:$q distX:$distX weight:$weight 
            #distZ:$distZ delta:$delta_fiedler gradient:$gradient"
        end
    end

    #alternative cost formula
    if false
        lambda = diag(vals)
        cost = log(prod(lambda[2:num_eigs]))-trace(L*fea*transpose(fea))/dim
    else
        cost = maximum(chosenSet[:,5])
    end
    variationRatio = mean(chosenSet[:,3])

    if minimum(chosenSet[:,3]) == 0
        return G, variationRatio, cost
    end

    for k in 1:length(chosenSet[:,1])
        add_edge!(G,chosenSet[k,1],chosenSet[k,2],chosenSet[k,3])
    end


    return G, variationRatio, cost
end

#FiedlerConstruct
function FiedlerConstruct(fea, samplePerc, edge_iter, sig, maxIterations, k, tol, num_node, dim)
    varRatio = Array{Float64}(undef,maxIterations)
    costVec = Array{Float64}(undef,maxIterations)

    #Generate Initial KNN

    #Intialize HNSW struct
    feaVec = [vec(fea[i,:]) for i = 1:num_node] #input should be a vector of vectors
    @info "$k NN HNSW:..."
    @time begin
        #hnsw = HierarchicalNSW(feaVec; efConstruction=100, M=45, ef=90)
        hnsw = HierarchicalNSW(feaVec; efConstruction=100, M=45, ef=100)
        #Add all data points into the graph
        #Optionally pass a subset of the indices in data to partially construct the graph
        add_to_graph!(hnsw)

        # Find k+1 (approximate) nearest neighbors for each of the queries; first result is usually itself
        # results (# edges) are slightly different each time
        idxs, dists = knn_search(hnsw, feaVec, k+1)


        e = undirectedEdges(idxs,dists)
        #Normalize edge weights by w = 1/dist^2
        we = [Pair(e[i][1],1/(e[i][2]^2)) for i = 1:length(e)]
        #Store as sparse graph, ~14000-14005 nodes for USPS dataset
        G = SimpleWeightedGraph(num_node)
        for i in we
            add_edge!(G,i[1][1],i[1][2],i[2])
        end

    end
    @info "Processed Graph" G

    #MatrixMarket.mmwrite("JuliaL.mtx",L)

    @info "GRASPEL Iterations..."
    @time begin
        numIter = 0
        for i in 1:maxIterations
            numIter = i
            G, varR, cost = FiedlerAddEdges(G,fea,samplePerc,edge_iter,sig,tol)

            varRatio[i] = varR
            costVec[i] = cost

            if varR < tol
                break
            end

        end
    end
    return G, varRatio[1:numIter], costVec[1:numIter], numIter
end