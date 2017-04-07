function [f, df] = ZSL_ObjFunc_Wx(W_x_vec, num_Parts, c, dx, W_z, X, Z, Y, ZZ_t,  D_xzi, lambda1, lambda2, GPU_mode)

W_x = reshape(W_x_vec, [c, dx]); 

dp = dx / num_Parts; 
W_x_t = W_x'; 

XX_t = X * X'; 
XYZ_t = X * Y * Z'; 

%%%% precompute multplication
Wxt_Wz = W_x' * W_z; 
Wxt_Wz_Z = Wxt_Wz * Z; 

trace_sum = 0; 
for i = 1 : num_Parts
    trace_sum = trace_sum + trace( W_x_t((dp*(i-1)+1) : dp*(i),:) * W_z * full(D_xzi{i}) * W_z' * W_x_t((dp*(i-1)+1) : dp*(i),:)'); 
end

%%%% calculate loss
f = norm((X'* Wxt_Wz_Z - Y) ,'fro')^2 + lambda1 * norm( Wxt_Wz_Z ,'fro')^2  + lambda2 * trace_sum; 
if(GPU_mode)
    f = gather(f); 
end

%%%% calculate the derivative of W_x
term0 = W_z * ZZ_t * Wxt_Wz' * XX_t - 2 * W_z * XYZ_t'; 
term1 = lambda1 * W_z * ZZ_t * Wxt_Wz'; 
if(GPU_mode)
    term2 = gpuArray(zeros(dx, c)); 
else 
    term2 = zeros(dx, c);
end
for i = 1 : num_Parts
    term2((dp*(i-1)+1) : dp*(i), :) =   W_x_t((dp*(i-1)+1) : dp*(i),:)* W_z * full(D_xzi{i}) * W_z'; 
end
term2 = lambda2 * term2; 

dW_x = 2 * (term0 + term1 + term2');
df = reshape(dW_x, [c*dx,1]); 

if(GPU_mode)
    df = gather(df); 
end

end



