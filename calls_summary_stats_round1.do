********************************************************************************
*			-----  qualtrics: data-cleaning -----


* This do-file prepares and explores the data collected in the first round and second round of surveys

* Author: Marco Caporaletti
* Created on: 05.2024
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



	
	**********************************************************************



	
	**** Import player data ****


	** First round **
	* import list of numbers called - first round
	import excel using "data/working/called_numbers_pilot_09022024.xlsx", firstrow clear
	
	* merge with to-call list to get player data
	preserve
	import excel using "survey_materials/list_calls_09feb24.xls", firstrow clear
	tempfile to_call_0902
	save `to_call_0902', replace
	restore
	
	merge 1:1 reference_phone using `to_call_0902'
	keep if _merge == 3
	drop _merge
	
	tempfile calls_first_round
	save `calls_first_round'
	

	** Second round **

	* import list of called numbers - second round
	import excel using "data/working/called_numbers_pilot_13022024.xlsx", firstrow clear
	
	* merge with list of numbers to call to get player data 
	preserve
	import delimited using "survey_materials/list_calls_small_winners_13022024.csv", clear
	gen date = date(date_only, "DMY")
	drop date_only
	rename date date_only 
	format date_only %td
	tempfile to_call_1302
	save `to_call_1302', replace
	restore
	
	merge 1:1 reference_phone agent using `to_call_1302'
	unique groupid
	unique groupid if _merge == 3 // only 235 groups out of 1243 are represented --> For the future, give them a list of calls that is to be carried out completely
	keep if _merge == 3 // only keep actually called 
	drop _merge
	tab groupid winner 
	
	* count how many groups have a winner with no controls
	preserve
	collapse (sum) winner control, by (groupid)
	count if winner ==0 & control >0 & groupid != .
	count if groupid != .
	restore
	* 24 out of 235
	
	tempfile calls_second_round
	save `calls_second_round'
	


	
	**********************************************************************


	

	**** Import survey data ****	


	* merge with qualtrics data on sucsessful calls
	
	use "data/working/qualtrics/qualtrics_23mar.dta", clear
	merge 1:1 reference_phone using `calls_first_round', update


	* generate round and pick_up variables 

	gen round_calls = 1
	replace round_calls = 2 if _merge == 1
	gen pick_up = 0
	replace pick_up = 1 if _merge != 1 & _merge != 2
	drop _merge
	
	merge 1:1 reference_phone using `calls_second_round', update
	replace round_calls = 2 if round_calls == .
	replace agent = "Phoebe" if agent == "Phoebe1"
	tab round_calls agent
	replace pick_up = 0 if pick_up == .
	replace pick_up = 1 if _merge != 1 & _merge != 2

	* inspect pick_up rate by round

	tab pick_up round_calls
	drop _merge
	
	encode consent, gen(consent_id)
	
	gen pick_up_consent = 0
	replace pick_up_consent = 1 if pick_up == 1 & consent_id == 2
	
	tab pick_up round_calls, co
	tab consent_id round_calls, co
	tab pick_up_consent round_calls
	tab language round_calls, co

	* generate "usable" variable (picked up, consented, in English) and inspect it by round
	
	gen usable = (pick_up_consent & language == "English")
	tab usable round_calls, co
	tab winner usable, co
	tab winner

	* inspect amount won by expenditure
	
	summarize total_winnings if winner == 1
	tab winner, sum(expenditure_total)
	
	* inspect time span between play and survey start
	gen time_diff = td(10feb2024)-date_only
	sum time_diff
	tab time_diff
	
	* Define value labels for winner variable 

	label define winner_label 0 "Loser" 1 "Winner"
	label values winner winner_label



	
	**********************************************************************



	
	**** EDA / Summaries ****


	* Histogram of total_winnings for winner = 1 

	histogram total_winnings if winner==1 & usable ==1, percent title("Histogram of amount won - Round 1") xtitle("total amount won")
	
	* Histogram of expenditure_total by winner 

	histogram expenditure_total if usable ==1, by(winner) percent width(100) xtitle("total amount played (GHS)")
	
	* pick-up rate and pick-up rate by winner
	
	tab winner, sum(pick_up) nost
	tab round_calls, sum(pick_up) nost // pick-up around 30%
	tab round_calls if winner == 0, sum(pick_up) nost
	tab round_calls if winner == 1, sum(pick_up) nost // weirdly, pick-up is a bit lower for winners
	
	* repeat with consent
	
	tab winner, sum(pick_up_consent) nost
	tab round_calls, sum(pick_up_consent) nost // pick-up around 30%
	tab round_calls if winner == 0, sum(pick_up_consent) nost
	tab round_calls if winner == 1, sum(pick_up_consent) nost // weirdly, pick-up is a bit lower for winners
	
	* does amount won influence pick-up?
	
	replace total_winnings = 0 if winner == 0
	tab total_winnings round_calls
	reg pick_up total_winnings
	reg pick_up total_winnings if round_calls == 1
	reg pick_up total_winnings if round_calls == 2
	reg pick_up total_winnings if winner == 1
	
	reg pick_up_consent total_winnings
	reg pick_up_consent total_winnings if round_calls == 1
	reg pick_up_consent total_winnings if round_calls == 2
	reg pick_up_consent total_winnings if winner == 1
	
		// not statistically significant influence of winnings on pick-up rate
	
	* does winning influence openness? (i.e. proportion_answered)
	tab winner, sum(proportion_answered) // yes, but in unexpected direction! Winners answer less. (this is only for the last 178 calls where we introduced feedback from agents)
	
	tab round_calls language, r
	



	**********************************************************************



	
	**** Summary stats on transcriptions  ****

	
	* merge with transcriptions 

	merge 1:1 reference_phone using "data/transcriptions/full_transcriptions_with_controls_11apr2024.dta", keepusing(file_name transcriber transcription)
	gen transcription_words = wordcount(transcription)
	replace transcription_words = . if transcription == ""
	gen transcribed = 0
	replace transcribed = . if pick_up == 0
	replace transcribed = 1 if _merge == 3
	drop _merge 
	
	tab transcribed pick_up, co // roughly 30% transcribed
	tab transcribed winner, co // pct transcribed is balanced across winner
	
	* does winning influence word count?
	tab winner, sum(transcription_words). // 250 words more on average, but they get asked one more questions (technically 2, but they are almost always answered together). 

	

	
	
	
	
	