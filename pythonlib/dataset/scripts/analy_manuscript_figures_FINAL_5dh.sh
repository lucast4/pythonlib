#!/bin/bash -e

animal=$1
 
if [[ $animal == Diego ]]; then
  datelist=(240517 240521 240523 240730)
  
elif [[ $animal == Pancho ]]; then
  datelist=(240516 240521 240524) # GOOD - dates

else
  echo $animal
  echo "Error! Inputed non-existing animal" 1>&2
  exit 1
fi

for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript_figures_FINAL_5dh_${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript_figures_FINAL.py 5dh ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done

