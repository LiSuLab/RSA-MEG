function tests=test_io
tests=functiontests(localfunctions);
end
function testSTCKnownBytes(t)
% Independently specified STC bytes: tmin=-100ms, dt=10ms; IDs 0 and 42;
% two time samples, vertices fastest. This is not only a writer round trip.
f=[tempname,'.stc']; cleanup=onCleanup(@()delete(f));
fid=fopen(f,'wb','ieee-be'); fwrite(fid,[-100,10],'single');
fwrite(fid,[2,0,42,2],'uint32');fwrite(fid,[1,2,3,4],'single');fclose(fid);
s=rsa_meeg.io.read_stc(f);
verifyEqual(t,s.vertices_zero_based,[0;42]);verifyEqual(t,s.data,[1 3;2 4]);
verifyEqual(t,s.sample_times_s,[-.1,-.09],'AbsTol',1e-12);
verifyError(t,@()rsa_meeg.io.write_stc(f,[0;42],s.data,s.sample_times_s),'rsa_meeg:io:Exists');
rsa_meeg.io.write_stc(f,[0;42],s.data,s.sample_times_s,'Overwrite',true);
b=rsa_meeg.io.read_stc(f);verifyEqual(t,b.data,s.data);
end
function testSTCRejectAndSingle(t)
f=[tempname,'.stc']; cleanup=onCleanup(@()delete(f));
rsa_meeg.io.write_stc(f,5,2,-.2,'TimeStep',.01);
s=rsa_meeg.io.read_stc(f); verifyEqual(t,s.sample_times_s,-.2,'AbsTol',1e-10);
verifyError(t,@()rsa_meeg.io.write_stc(f,[1,1],[1;2],0,'TimeStep',.01),'rsa_meeg:io:Vertices');
fid=fopen(f,'ab');fwrite(fid,0,'uint8');fclose(fid);
verifyError(t,@()rsa_meeg.io.read_stc(f),'rsa_meeg:io:STC');
end
function testSurfaceAndLabel(t)
f=tempname;g=[tempname,'.label'];cleanup=onCleanup(@()delete_files({f,g}));
fid=fopen(f,'wb','ieee-be');fwrite(fid,[255 255 254],'uint8');fprintf(fid,'fixture\n\n');
fwrite(fid,[3 1],'int32');fwrite(fid,[0 0 0;1 0 0;0 1 0]','single');fwrite(fid,[0 1 2],'int32');fclose(fid);
s=rsa_meeg.io.read_surface(f);verifyEqual(t,s.triangles,[1 2 3]);verifyEqual(t,s.positions_mm(2,:),[1 0 0]);
fid=fopen(g,'wt');fprintf(fid,'# test\n2\n0 0 0 0 1\n42 1 2 3 0\n');fclose(fid);
l=rsa_meeg.io.read_label(g);verifyEqual(t,l.vertices_zero_based,[0;42]);
end
function testSTCHemispheres(t)
folder=tempname;mkdir(folder);cleanup=onCleanup(@()rmdir(folder,'s')); files=cell(2,2);
for o=1:2
    for h=1:2
        files{o,h}=fullfile(folder,sprintf('o%d-h%d.stc',o,h));
        rsa_meeg.io.write_stc(files{o,h},[0 20],[1 2;3 4]*o*h,[0,.01]);
    end
end
ds=rsa_meeg.io.from_stc(files,{'a','b'},'Hemisphere',{'lh','rh'});
verifyEqual(t,size(ds.data),[2 4 2]);
verifyEqual(t,string(ds.feature_ids),["lh:0","lh:20","rh:0","rh:20"]);
verifyEqual(t,reshape(ds.data(2,3:4,:),2,2),[4 8;12 16]);
verifyError(t,@()rsa_meeg.io.from_stc(files,{'a','b'}),'rsa_meeg:io:Hemisphere');
end
function testEEGLABAndFieldTrip(t)
x=reshape(1:24,2,3,4); EEG=struct('data',x,'times',[-10 0 10]);
verifyError(t,@()rsa_meeg.io.from_eeglab(struct('data',zeros(2,3),'times',[-10 0 10]),1),'rsa_meeg:io:EEGLAB');
ds=rsa_meeg.io.from_eeglab(EEG,{'a','b','a','b'},'FoldIds',[1 1 2 2],'Reference','average');
verifyEqual(t,ds.data,permute(x,[3 1 2]));verifyEqual(t,ds.sample_times_s,[-.01 0 .01]);
ft.label={'a','b'};ft.trial={x(:,:,1),x(:,:,2)};ft.time={[-.01 0 .01],[-.01 0 .01]};
d=rsa_meeg.io.from_fieldtrip(ft,[1 2],'FeatureTypes',{'mag','mag'});
verifyEqual(t,squeeze(d.data(1,:,:)),x(:,:,1));
ft.time{2}(3)=.02;
verifyError(t,@()rsa_meeg.io.from_fieldtrip(ft,[1 2]),'rsa_meeg:io:Alignment');
end
function delete_files(files)
for k=1:numel(files),if isfile(files{k}),delete(files{k});end,end
end
