using Colors, Plots, LinearAlgebra, MAT
vars = matread(raw"C:\Users\jacec\Desktop\code\code\USPS.mat")
plotUSPSfea(A::Matrix,index::Int) = plot(Gray.(transpose(reshape(A[index,:],(16,16)))))
p = Vector{typeof(plot())}(undef,5)
for i = 1:5
    #display(plotUSPSfea(vars["fea"],i))
    p[i] = plotUSPSfea(vars["fea"],i)
end
plot(p...)