In the folder Files_provided_by_the_teacher/test_cases_in_out we have some inputs and their expected output.

To run, just download the folder we're in and run something like this on your terminal:
	
 	python main.py < Files_provided_by_the_teacher/test_cases_in_out/case2.in

In this case, you're gonna run the code given the input in case2.in. You can compare this to the output expected in case2.out and see that is very close. 

When you run the code for a case, it will also generate a plot for:
- The input image
- The Fourier Transform of the input image
- The filter generated for the case
- The restored image given the filter and the input image

The number that the code prints it's the RMSE between the restored image and the reference image that is given for each case in the folder Files_provided_by_the_teacher/test_cases_reference 
