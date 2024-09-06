********************************************************************************
*			-----  qualtrics: data-cleaning -----


* This do-file prepares an excel file for splitting the different questions 
* in the cleaned transcriptions from the Qualtrics Transcriptions based on randomization. 

* Things to update before running: relevant dates and weeks (marked with x)

* input: data/transcriptions/full_transcriptions_with_controls_`day'`month'2024.dta

* output: data/transcriptions/transcriptions_all_split_`day'`month'2024
* excel file in wide shape with (among others) following variables:
* reference_phone, question_id, question_text, question_answer (empty), (full) transcription,
* winner, total_winnings

* This version covers transcriptions done until the 11 of April: all of round 1


* Author: Marco Caporaletti
* Created on: 23.04.2024
*
********************************************************************************
	
	clear all
	
	dis "`c(username)'"


	
	* set global variable path *

	*********************************************************************
	
	if "`c(username)'" == "marcocaporaletti"{
	global path "/Users/marcocaporaletti/Dropbox/Experiment_gambling/Voices_of_Ghana"
	}
	
	* add paths for other team members *
	

	cd $path
	* for saving the dataset with a new name
	* x // is this the correct date?
	local day 11
	local month apr
	
	*********************************************************************
	

	* Import transcriptions
	use "data/transcriptions_qualtrics/full_transcriptions_with_controls_`day'`month'2024.dta", clear
	compress file_name
	
	* Import transcriptions previously selected to split by hand to flag them
	preserve
	use "data/transcriptions_qualtrics/full_transcriptions_selected_23mar2024.dta", clear
	compress file_name
	keep file_name selected
	tempfile selected
	save `selected'
	restore
	
	merge 1:1 file_name using `selected'
	tab selected _merge
	replace selected = 0 if selected == .
	keep if selected == 0
	drop selected _merge
	
	
	* Expand data based on winner variable 
	expand 8 if winner == 0
	expand 9 if winner == 1
	sort winner file_name

	drop transcription_2
	gen question_id = "self_intro"
	gen question_text = "To begin, we would like to know you better. Can you tell me a little bit more about yourself and your family?"
	gen question_answer = ""
	ren transcription full_transcription
	
	order question_id question_text question_answer randomizedquestionsmoney_do randomizedquestionshealth_do, after(file_name)
	order winner total_winnings, after(full_transcription)
	
	* fill out question_text and question_id based on randomization
	by winner file_name: gen question_num = _n
	order question_num, after(file_name)
	replace question_num = question_num - 1
	
	replace question_id = "demographics" if question_num == 0
	replace question_text = "" if question_num == 0
	
	replace question_id = "money_daily" if question_num == 2
	replace question_text = "How do you organize and keep track of your finances from day to day? Please tell me about any specific methods and tools you might use in this process." if question_num == 2
	
	replace question_id = randomizedquestionsmoney_do if question_num == 3
	replace question_text = "How does your financial situation now compare to a year ago?" if question_id == "financial_situation"
	replace question_text = "What is your personal experience with saving money?" if question_id == "saving"
	replace question_text = "Do you have any debts? If yes, why did you take a loan?" if question_id == "debts"
	replace question_text = "How do you learn about the best ways to manage your money?" if question_id == "management_learn"

	replace question_id = "health_general" if question_num == 4
	replace question_text = "How would you describe your and your family's health?" if question_num == 4

	replace question_id = randomizedquestionshealth_do if question_num == 5
	replace question_text = "If you think about the past ten days, what are some things that make you happy? What are some things that make you sad?" if question_id == "happiness"
	replace question_text = "How does your current stress level compare to same time last year?" if question_id == "stress"
	replace question_text = "Do you consume alcohol? If yes, how often, and why?" if question_id == "alcohol"

	replace question_id = "suggestions" if question_num == 6
	replace question_text = "How can we design a better and more popular lottery? Do you have any suggestions for us?" if question_num == 6

	replace question_id = "conclusion" if question_num == 7
	replace question_text = "" if question_num == 7
	
	replace question_id = "impact" if question_num == 7 & winner == 1
	replace question_text = "In what way has winning the raffle impacted your life? How have you spent the money, or how do you plan to spend it?" if question_num == 7 & winner == 1
	
	replace question_id = "conclusion" if question_num == 8
	replace question_text = "" if question_num == 8
	
	export excel using "data/transcriptions_all/transcriptions_all_split_`day'`month'2024", replace firstrow(variables)
	*/
	