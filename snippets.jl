e1 = [e[i][1][1] for i = 1:length(e)]
e2 = [e[i][1][2] for i = 1:length(e)]

for i in 1:length(feaVec)
    if !(i in e1 || i in e2)
        @info i
    end
end

Ld = diag(laplacian_matrix(G))
for i in 1:length(Ld)
    if Ld[i] == 1e-6
        @info i
    end
end

[(idxs[i],dists[i]) for i in 1:length(idxs)]