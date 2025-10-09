#!/bin/bash -e


##################
animal=Diego
datelist=({230110..230119}) # gridlinecircle3 - LC and CL
# datelist=({230120..230129}) # gridlinecircle3 - lolli
# datelist=(230110 230111 230112) # Missed
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript2syntax_figures_1_gen.sh-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript2_figures.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done

sleep 2h

animal=Pancho
start="220831"  # YYMMDD
end="220909"    # YYMMDD
datelist=()
current="$start"
while [ "$(date -d "20$current" +%Y%m%d)" -le "$(date -d "20$end" +%Y%m%d)" ]; do
    datelist+=("$(date -d "20$current" +%y%m%d)")
    current=$(date -d "20$current +1 day" +%y%m%d)
done
echo "${datelist[@]}"
# datelist=({220831..220909}) # gridlinecircle3 - All dates
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript2syntax_figures_1_gen.sh-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript2_figures.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done


### Pancho -- gridlinecircle2
animal=Pancho
start="210820"  # YYMMDD
end="210820"    # YYMMDD
datelist=()
current="$start"
while [ "$(date -d "20$current" +%Y%m%d)" -le "$(date -d "20$end" +%Y%m%d)" ]; do
    datelist+=("$(date -d "20$current" +%y%m%d)")
    current=$(date -d "20$current +1 day" +%y%m%d)
done
echo "${datelist[@]}"
# datelist=({220831..220909}) # gridlinecircle3 - All dates
for date1 in "${datelist[@]}"
do
  logfile="../logs/analy_manuscript2syntax_figures_1_gen.sh-${animal}_${date1}.txt"
  touch ${logfile}
  echo ${logfile}
  python analy_manuscript2_figures.py ${animal} ${date1} 2>&1 | tee ${logfile} &
  sleep 10s
done
