function [ tDemo, xDemo ] = high5_joint2( dt )

% K=200,D=50, Scott's differentiation
% K=500,D=75, Scott's differentiation
% K=400,D=30, new differentiation

xDemo=[ 77.73626373626374
 77.38461538461539
 76.94505494505495
 76.50549450549451
 75.8021978021978
 75.18681318681318
 74.3076923076923
 73.42857142857143
 72.72527472527473
 71.84615384615384
 70.96703296703296
 69.82417582417582
 68.85714285714286
 67.89010989010988
 66.83516483516483
 65.86813186813187
 64.81318681318682
 63.582417582417584
 62.61538461538461
 61.032967032967036
 59.8021978021978
 58.48351648351648
 57.16483516483517
 55.67032967032967
 54.43956043956044
 52.94505494505494
 51.45054945054945
 50.13186813186813
 48.37362637362637
 47.05494505494506
 45.56043956043956
 44.15384615384615
 42.83516483516483
 41.252747252747255
 39.934065934065934
 38.43956043956044
 36.94505494505494
 35.62637362637363
 34.21978021978022
 32.637362637362635
 31.318681318681318
 30.175824175824175
 29.032967032967033
 28.065934065934066
 27.186813186813186
 26.307692307692307
 25.604395604395606
 25.076923076923077
 24.46153846153846
 24.10989010989011
 23.75824175824176
 23.582417582417584
 23.406593406593405
 23.318681318681318
 23.23076923076923
 23.054945054945055
 22.87912087912088
 22.703296703296704
 22.52747252747253
 22.263736263736263
 22.087912087912088
 21.912087912087912
 21.736263736263737
 21.64835164835165
 21.384615384615383
 21.208791208791208
 21.208791208791208
 21.208791208791208
 21.208791208791208
 21.47252747252747
 21.824175824175825
 22.087912087912088
 22.263736263736263
 22.439560439560438
 22.439560439560438
 22.52747252747253
 22.615384615384617
 22.52747252747253
 22.52747252747253
 22.439560439560438
 22.35164835164835
 22.263736263736263
 22.175824175824175
 22.087912087912088
 22.087912087912088
 22.0
 21.912087912087912
 21.912087912087912
 21.912087912087912
 21.912087912087912
 21.912087912087912
 21.912087912087912
 21.912087912087912
 21.912087912087912
 21.912087912087912
 21.824175824175825
 21.824175824175825
 21.912087912087912
 21.912087912087912
 22.0
 22.087912087912088
 22.175824175824175
 22.175824175824175
 22.263736263736263
 22.35164835164835
 22.35164835164835
 22.35164835164835
 22.35164835164835
 22.439560439560438
 22.439560439560438
 22.439560439560438
 22.35164835164835
 22.439560439560438
 22.35164835164835
 22.439560439560438
 22.35164835164835
 22.439560439560438]';

tDemo=0:dt:dt*(size(xDemo,2)-1);

end
