function [f, df] = ZSL_ObjFunc_Wz(W_z_vec, num_Parts,  c, dx, dz, W_x, X, Z, Y, ZZ_t,  D_xzi, lambda1, lambda2, GPU_mode)
 
W_z = reshape(W_z_vec, [c, dz]); 

dp = dx / num_Parts; 
W_x_t = W_x'; 

XX_t = X * X'; 
XYZ_t = X * Y * Z'; 

%%%% precompute multplication
Wxt_Wz = W_x' * W_z; 
Wxt_Wz_Z = Wxt_Wz * Z;

trace_sum = 0; 
for i = 1:num_Parts
    trace_sum = trace_sum + trace( W_x_t((dp*(i-1)+1) : dp*(i),:) * W_z * full(D_xzi{i}) * W_z' * W_x_t((dp*(i-1)+1):dp*(i),:)'); 
end

%%%% calculate loss
f =  norm( (X'* Wxt_Wz_Z - Y) ,'fro')^2 + lambda1 * norm( Wxt_Wz_Z ,'fro')^2 + lambda2 * trace_sum; 
if(GPU_mode)
    f = gather(f); 
end
%%%% calculate the derivative of W_z
term0 = W_x * XX_t * Wxt_Wz * ZZ_t - W_x * XYZ_t; 
term1 = lambda1 * W_x * Wxt_Wz * ZZ_t ; 
if(GPU_mode)
    term2 = gpuArray(zeros(c, dz)); 
else 
    term2 = zeros(c, dz);
end

for i = 1:num_Parts
    term2 = term2 + W_x_t((dp*(i-1)+1) : dp*(i),:)'*  W_x_t((dp*(i-1)+1) : dp*(i),:) * W_z * full(D_xzi{i}); 
end
term2 = term2 * lambda2; 
dW_z = 2 * (term0 + term1 + term2);
df = reshape(dW_z, [c*dz,1]); 
if(GPU_mode)
    df = gather(df); 
end

end



