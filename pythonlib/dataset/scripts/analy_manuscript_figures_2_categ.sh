#!/bin/bash -e


##################
animal=Diego
# datelist=(240517 240521 240523 240730) # Switching
# datelist=(240515 240517 240523 240731 240801 240802) # Smooth
datelist=(240731 240801 240802) # Smooth, (failed, rerunning)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript_figures_2_categ-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript_figures.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done

animal=Pancho
# datelist=(240516 240521 240524) # Switching
# datelist=(240516 240521 240524 240801 240802) # Smooth
datelist=(240801 240802) # Smooth, (failed, rerunning)
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript_figures_2_categ-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript_figures.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done
