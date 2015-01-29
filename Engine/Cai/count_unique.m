function [unique_values, unique_count] = count_unique(input)
% Given a 2-d matrix `input`, count_unique returns two values:
%
% `unique_values`: A list of the unique values in `input`
%                  (the same list as would be returned by
%                  running unique(input))
%
% `unique_count`:  A list of the number of occurences of each
%                  unique value in the corresponding position
%                  in `unique_values`.
%
% CW 2015-01

    % validation checks
    if ~ismatrix(input)
        error('Input matrix must be 2-d.');
    end

    % list of unique entries from the input
    unique_values = unique(input);
    unique_value_count = length(unique_values);
    % reshaped to stretch down the third dimension
    tall_unique_values = reshape(unique_values, [1,1, unique_value_count]);

    % a copy of the input for each unique value
    stacked_a = repmat(input, 1, 1, unique_value_count);

    % a matrix which has a one input-sized page of each unique value of
    % the input
    stacked_c = repmat(tall_unique_values, size(input));

    % a logical matrix which, for each page, shows the positions of each
    % unique values of the input
    unique_positions = (stacked_a == stacked_c);

    % the number of each occurence of unique elements of the input
    unique_count = reshape(sum(sum(unique_positions, 1), 2), [unique_value_count, 1, 1]);

end%function
