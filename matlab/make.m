% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
% Credits: make.m taken from LIBLINEAR implementation
% Modified by Nemanja Djuric to check for Matlab 32- or 64-bit versions
function make()

try
	Type = ver;
	if (strcmp(Type(1).Name, 'Octave') == 1)
		% This part is for OCTAVE
		disp('make process started, please be patient as it might take about a minute.');
		disp('Compiling libsvmread.c ...');
		mex libsvmread.c
		disp('Compiling libsvmwrite.c ...');
		mex libsvmwrite.c
		disp('Compiling budgetedsvm_train.cpp ...');
		mex budgetedsvm_train.cpp ../src/budgetedSVM.cpp ../src/mm_algs.cpp ../src/bsgd.cpp ../src/llsvm.cpp budgetedSVM_matlab.cpp
		disp('Compiling budgetedsvm_predict.cpp ...');
		mex budgetedsvm_predict.cpp ../src/budgetedSVM.cpp ../src/mm_algs.cpp ../src/bsgd.cpp ../src/llsvm.cpp budgetedSVM_matlab.cpp
	else
		% This part is for MATLAB
		disp('make process started, please be patient as it might take about a minute.');
        if (is_64_bit_version())
            % Add -largeArrayDims on 64-bit machines of MATLAB
			disp('Compiling libsvmread.c ...');
            mex CFLAGS='\$CFLAGS -std=c99' -largeArrayDims libsvmread.c
			disp('Compiling libsvmwrite.c ...');
            mex CFLAGS='\$CFLAGS -std=c99' -largeArrayDims libsvmwrite.c
			disp('Compiling budgetedsvm_train.cpp ...');
            mex CFLAGS='\$CFLAGS -std=c99' -largeArrayDims budgetedsvm_train.cpp ../src/budgetedSVM.cpp ../src/mm_algs.cpp ../src/bsgd.cpp ../src/llsvm.cpp budgetedSVM_matlab.cpp
			disp('Compiling budgetedsvm_predict.cpp ...');
            mex CFLAGS='\$CFLAGS -std=c99' -largeArrayDims budgetedsvm_predict.cpp ../src/budgetedSVM.cpp ../src/mm_algs.cpp ../src/bsgd.cpp ../src/llsvm.cpp budgetedSVM_matlab.cpp
        else		
			disp('Compiling libsvmread.c ...');
            mex CFLAGS='\$CFLAGS -std=c99' libsvmread.c
			disp('Compiling libsvmwrite.c ...');
            mex CFLAGS='\$CFLAGS -std=c99' libsvmwrite.c
			disp('Compiling budgetedsvm_train.cpp ...');
            mex CFLAGS='\$CFLAGS -std=c99' budgetedsvm_train.cpp ../src/budgetedSVM.cpp ../src/mm_algs.cpp ../src/bsgd.cpp ../src/llsvm.cpp budgetedSVM_matlab.cpp
			disp('Compiling budgetedsvm_predict.cpp ...');
            mex CFLAGS='\$CFLAGS -std=c99' budgetedsvm_predict.cpp ../src/budgetedSVM.cpp ../src/mm_algs.cpp ../src/bsgd.cpp ../src/llsvm.cpp budgetedSVM_matlab.cpp
        end;
	end
catch e
	fprintf('make.m failed. Please check README.txt for more detailed instructions.\n');
end;


function is_64b = is_64_bit_version()
ext_string = mexext();
if (str2double(ext_string(end - 1 : end)) == 32)
    is_64b = false;
elseif (str2double(ext_string(end - 1 : end)) == 64)
    is_64b = true;
else
    error('Could not verify if Matlab is 32-bit or 64-bit version. Please check and modify make.m to compile the source codes on your platform.');
end;
