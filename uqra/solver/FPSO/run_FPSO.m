function y = run_FPSO(Hs,Tp,seed)
    global Diag_surge N N_LF Snn TFv dw dw_LF w w_LF RAO wmin wmax Hx

    load freq_data
    load seeds;
    %% 
    FD_FPSO_LF

    %%
    % ---
    % QTF
    % ---

    Hx=(-w_LF.^2*M(1,1)+1i*w_LF*B(1,1)+K(1,1)).^-1;
    Hx(N)=0;
    xinput(1,1:2)=[Hs, Tp];
    rng(seed);
    xinput(1,3:1922)=normrnd(0,1,1,960*2);
    y=Glimitmax(xinput); 
end