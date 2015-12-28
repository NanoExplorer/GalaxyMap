#This isn't really a shell script... I wouldn't recommend running it. Just use it to stage commands and copy-paste them

python survey_perturb.py surveys/100s/COMPOSITE-1-survey.dat surveys/perturbed/COMPOSITE-1-perturbed-0.1-{}.dat 0.1 100
python survey_perturb.py surveys/100s/COMPOSITE-1-survey.dat surveys/perturbed/COMPOSITE-1-perturbed-0.2-{}.dat 0.2 100
python survey_perturb.py surveys/100s/COMPOSITE-69-survey.dat surveys/perturbed/COMPOSITE-29-perturbed-0.1-{}.dat 0.1 100
python survey_perturb.py surveys/100s/COMPOSITE-69-survey.dat surveys/perturbed/COMPOSITE-29-perturbed-0.2-{}.dat 0.2 100
python survey_perturb.py surveys/100s/CF2-gal-51-survey.dat surveys/perturbed/CF2-gal-51-perturbed-0.1-{}.dat 0.1 100
python survey_perturb.py surveys/100s/CF2-gal-51-survey.dat surveys/perturbed/CF2-gal-51-perturbed-0.2-{}.dat 0.2 100
python survey_perturb.py surveys/100s/CF2-gal-62-survey.dat surveys/perturbed/CF2-gal-62-perturbed-0.1-{}.dat 0.1 100
python survey_perturb.py surveys/100s/CF2-gal-62-survey.dat surveys/perturbed/CF2-gal-62-perturbed-0.2-{}.dat 0.2 100
python survey_perturb.py surveys/100s/CF2-group-83-survey.dat surveys/perturbed/CF2-group-83-perturbed-0.1-{}.dat 0.1 100
python survey_perturb.py surveys/100s/CF2-group-83-survey.dat surveys/perturbed/CF2-group-83-perturbed-0.2-{}.dat 0.2 100
python survey_perturb.py surveys/100s/CF2-group-38-survey.dat surveys/perturbed/CF2-group-38-perturbed-0.1-{}.dat 0.1 100
python survey_perturb.py surveys/100s/CF2-group-38-survey.dat surveys/perturbed/CF2-group-38-perturbed-0.2-{}.dat 0.2 100




python survey_perturb.py surveys/100s/COMPOSITE-1-survey.dat surveys/perturbed/COMPOSITE-1-p-0.1-naive-{}.dat 0.1 100 -n
python survey_perturb.py surveys/100s/COMPOSITE-1-survey.dat surveys/perturbed/COMPOSITE-1-p-0.2-naive-{}.dat 0.2 100 -n
python survey_perturb.py surveys/100s/COMPOSITE-1-survey.dat surveys/perturbed/COMPOSITE-1-p-0.1-notmod-{}.dat 0.1 100 -d
python survey_perturb.py surveys/100s/COMPOSITE-1-survey.dat surveys/perturbed/COMPOSITE-1-p-0.2-notmod-{}.dat 0.2 100 -d
python survey_perturb.py surveys/100s/COMPOSITE-69-survey.dat surveys/perturbed/COMPOSITE-29-p-0.1-naive-{}.dat 0.1 100 -n
python survey_perturb.py surveys/100s/COMPOSITE-69-survey.dat surveys/perturbed/COMPOSITE-29-p-0.2-naive-{}.dat 0.2 100 -n
python survey_perturb.py surveys/100s/COMPOSITE-69-survey.dat surveys/perturbed/COMPOSITE-29-p-0.1-notmod-{}.dat 0.1 100 -d
python survey_perturb.py surveys/100s/COMPOSITE-69-survey.dat surveys/perturbed/COMPOSITE-29-p-0.2-notmod-{}.dat 0.2 100 -d




