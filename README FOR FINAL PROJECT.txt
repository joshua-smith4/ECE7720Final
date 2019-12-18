Windows instructions: 
Open a powershell window in this folder, then run
python ./FILENAME.py

Linux instruction
Open a terminal in this folder, then run
python ./FILENAME.py

gen_adv_ex_serial_FGSM.py
runs the FGSM algorithm with no modifications. 
Writes its data to the FGSM.txt

gen_adv_ex_serial_wm_after.py
Runs the FGSM algorithm with the watermark applied after the FGSM 
has been applied. This file writes it output to the test.txt file.

gen_adv_ex_serial_wm_before.py
Runs the FGSM algorithm with the watermark applied before the FGSM 
has been applied. This file writes it output to the test_before.txt file.
