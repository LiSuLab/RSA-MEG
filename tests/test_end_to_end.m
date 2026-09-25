function tests=test_end_to_end
tests=functiontests(localfunctions);
end
function testModalitiesAndRNG(t)
state=rng;
for modality={'eeg','mag','grad','source'}
    out=demo_synthetic(modality{1},false);
    verifySize(t,out.fit.values,[12 15]);
    verifyTrue(t,all(isfinite(out.rdm.values(:))));
    signal=out.fit.windows.center_s>.065&out.fit.windows.center_s<.135;
    sig=reshape(out.inference.significant,size(out.fit.values));
    verifyTrue(t,any(sig(:,signal),'all'));
    verifyEqual(t,out.inference.n_permutations,256);
end
verifyEqual(t,rng,state);
end
