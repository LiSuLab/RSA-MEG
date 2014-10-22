function d = howFar(a,b,varargin)

if ~isequal(size(a), size(b))
    error('Can only find distance if vectors are of the same size.');
end%if

nData = numel(a);

if numel(varargin) == 0
    distanceMeasure = 'correlation';
elseif numel(varargin) == 1 && ischar(varargin{1})
    distanceMeasure = varargin{1};
    
    switch distanceMeasure
        case 'euclidean'
            error('Not yet implemented.');
%             dsq = zeros(n-i,1,outClass);
%             for q = 1:p
%                 dsq = dsq + (X(i,q) - X((i+1):n,q)).^2;
%             end
%             Y(k:(k+n-i-1)) = sqrt(dsq);
        case 'seuclidean'
            error('Not yet implemented.');
%             wgts = additionalArg;
%             dsq = zeros(n-i,1,outClass);
%             for q = 1:p
%                 dsq = dsq + wgts(q) .* (X(i,q) - X((i+1):n,q)).^2;
%             end
%             Y(k:(k+n-i-1)) = sqrt(dsq);
        case 'cityblock'
            error('Not yet implemented.');
%              d = zeros(n-i,1,outClass);
%             for q = 1:p
%                 d = d + abs(X(i,q) - X((i+1):n,q));
%             end
%             Y(k:(k+n-i-1)) = d;
        case 'mahalanobis'
            error('Not yet implemented.');
%             invcov = additionalArg;
%             del = repmat(X(i,:),n-i,1) - X((i+1):n,:);
%             dsq = sum((del*invcov).*del,2);
%             Y(k:(k+n-i-1)) = sqrt(dsq);
        case 'minkowski'
            error('Not yet implemented.');
%              expon = additionalArg;
%             dpow = zeros(n-i,1,outClass);
%             for q = 1:p
%                 dpow = dpow + abs(X(i,q) - X((i+1):n,q)).^expon;
%             end
%             Y(k:(k+n-i-1)) = dpow .^ (1./expon);
        case 'cosine'
            error('Not yet implemented.');
%               % This assumes that data have been appropriately preprocessed
%             d = zeros(n-i,1,outClass);
%             for q = 1:p
%                 d = d + (X(i,q).*X((i+1):n,q));
%             end
%             Y(k:(k+n-i-1)) = 1 - d;
        case 'correlation'
            d = 1 - corr(a, b, 'type', 'Pearson');
        case 'spearman'
            d = 1 - corr(a, b, 'type', 'Spearman');
        case 'hamming'
            error('Not yet implemented.');
%              nesum = zeros(n-i,1,outClass);
%             for q = 1:p
%                 nesum = nesum + (X(i,q) ~= X((i+1):n,q));
%             end
%             nesum(nans(i) | nans((i+1):n)) = NaN;
%             Y(k:(k+n-i-1)) = nesum ./ p;
        case 'jaccard'
            error('Not yet implemented.');
%               nzsum = zeros(n-i,1,outClass);
%             nesum = zeros(n-i,1,outClass);
%             for q = 1:p
%                 nz = (X(i,q) ~= 0 | X((i+1):n,q) ~= 0);
%                 ne = (X(i,q) ~= X((i+1):n,q));
%                 nzsum = nzsum + nz;
%                 nesum = nesum + (nz & ne);
%             end
%             nesum(nans(i) | nans((i+1):n)) = NaN;
%             Y(k:(k+n-i-1)) = nesum ./ nzsum;
        case 'chebychev'
            error('Not yet implemented.');
%             dmax = zeros(n-i,1,outClass);
%             for q = 1:p
%                 dmax = max(dmax, abs(X(i,q) - X((i+1):n,q)));
%             end
%             dmax(nans(i) | nans((i+1):n)) = NaN;
%             Y(k:(k+n-i-1)) = dmax;
        otherwise
            error('Third argument must be a valid distance measure.')
    end
else
    error('Wrong number of arguments (please use 2 or 3).');
end