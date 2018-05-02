# Compare your continuous ROC curves with other methods
# At terminal: gnuplot contROC_unpub.p
set terminal png size 1280, 1200 enhanced font 'Verdana,18'
set size 1,1
set xtics 500
set ytics 0.1
set grid
set ylabel "True positive rate"
set xlabel "False positive"
set xr [0:2000]
set yr [0:1.0]
set key below
set output "contROC_unpub.png"
plot   "rocCurves/PingAnAI-ContROC.txt" using 2:1 title 'PingAnAI' with linespoints pointinterval 80 lw 2 , \
"rocCurves/NanYunTech-ContROC.txt" using 2:1 title 'NanYunTech' with linespoints pointinterval 80 lw 2 , \
"rocCurves/Glasssix-ContROC.txt" using 2:1 title 'Glasssix' with linespoints pointinterval 80 lw 2 , \
"rocCurves/ReadSense_ContROC_v2.txt" using 2:1 title 'ReadSense-v2' with linespoints pointinterval 80 lw 2 , \
"rocCurves/ReadSense_ContROC_v1.txt" using 2:1 title 'ReadSense-v1' with linespoints pointinterval 80 lw 2 , \
"rocCurves/Authenmetric-CBSR-v1_ContROC.txt" using 2:1 title 'Authenmetric-v2' with linespoints pointinterval 80 lw 2 , \
"rocCurves/wisesoft_ContROC_v1.txt" using 2:1 title 'Wisesoft' with linespoints pointinterval 80 lw 2 , \
"rocCurves/TencentAI_FDDB_ContROC.txt" using 2:1 title 'TencentAI.v1' with linespoints pointinterval 80 lw 2 , \
"rocCurves/MT-Face-v3-ContROC.txt" using 2:1 title 'MT-Face-v3' with linespoints pointinterval 80 lw 2 , \
"rocCurves/HisignContROC_v2.txt" using 2:1 title 'Hisign-v2' with linespoints pointinterval 80 lw 2 , \
"rocCurves/THU-ContROC.txt" using 2:1 title 'THU CV-AI Lab' with linespoints pointinterval 80 lw 2 , \
"rocCurves/Uniview-v1-ContROC.txt" using 2:1 title 'Uniview' with linespoints pointinterval 80 lw 2 , \
"rocCurves/BAIDU-IDL-cont-v4.txt" using 2:1 title 'Baidu-IDL-v4' with linespoints pointinterval 80 lw 2 , \
"rocCurves/M-Face-tempContROC.txt" using 2:1 title 'MT-Face-V2' with linespoints pointinterval 80 lw 2 , \
"rocCurves/Emotibot-ContROC.txt" using 2:1 title 'Emotibot' with linespoints pointinterval 80 lw 2, \
"rocCurves/Daream_Cont-v1.txt" using 2:1 title 'Daream' with lines lw 2 , \
"rocCurves/M-FaceContROC.txt" using 2:1 title 'MT-Face' with lines lw 2 , \
"rocCurves/Authenmetric_ContROC.txt" using 2:1 title 'Authenmetric-v1' with lines lw 2 , \
"rocCurves/MS-FRCNN-ContROC.txt" using 2:1 title 'MS-FRCNN' with lines lw 2 , \
"rocCurves/ColorRecoContROC.txt" using 2:1 title 'ColorReco' with lines lw 2 , \
"rocCurves/Faceall_v2_ContROC.txt" using 2:1 title 'Faceall-v2' with lines lw 2 , \
"rocCurves/Faceall_ContROC.txt" using 2:1 title 'Faceall' with lines lw 2 , \
 "rocCurves/CW-DNA-v1ContROC.txt" using 2:1 title 'CW-DNA-v1' with lines lw 2 , \
 "rocCurves/CW-DNA-v2ContROC.txt" using 2:1 title 'CW-DNA-v2' with lines lw 2 , \
 "rocCurves/BAIDU-IDL-cont-v3.txt" using 2:1 title 'Baidu-IDL-v3' with lines lw 2 , \
 "rocCurves/360-NUS-ContROC.txt" using 2:1 title '360-NUS' with lines lw 2 , \
 "rocCurves/Yunnan-ContROC.txt" using 2:1 title 'Yunnan' with lines lw 2 , \
 "rocCurves/BAIDU-IDL-cont-v2.txt" using 2:1 title 'Baidu-IDL-v2' with lines lw 2 , \
 "rocCurves/PCI-FACEContROC.txt" using 2:1 title 'PCI-face' with lines lw 2 , \
 "rocCurves/Linkface-ContROC_v3.txt" using 2:1 title 'Linkface' with lines lw 2 , \
 "rocCurves/BAIDU-IDL-cont.txt" using 2:1 title 'Baidu-IDL' with lines lw 2 , \
 "rocCurves/Uni-Ubi_ContROC-2.txt" using 2:1 title 'Uni-Ubi-2' with lines lw 2 , \
 "rocCurves/LandmarkbasedContROC.txt" using 2:1 title 'IMC Lab' with lines lw 2 , \
 "rocCurves/Uni-Ubi_ContROC.txt" using 2:1 title 'Uni-Ubi' with lines lw 2 , \
 "rocCurves/Eyedea_ContROC.txt" using 2:1 title 'EyedeaRecognition' with lines lw 2 , \
 "rocCurves/StradVision-ContROC.txt" using 2:1 title 'StradVision' with lines lw 2 , \
 "rocCurves/tencent_ContROC.txt" using 2:1 title 'Tencent-BestImage' with lines lw 2 , \
 "rocCurves/Shenzhen2014-ContROC.txt" using 2:1 title 'Shenzhen2014' with lines lw 2 , \
 "rocCurves/tvplay.cn-ContROC.txt" using 2:1 title 'TVplay' with lines lw 2 , \
 "rocCurves/HisignContROC.txt" using 2:1 title 'Hisign' with lines lw 2 , \
 "rocCurves/Face++ContROC.txt" using 2:1 title 'Face++' with lines lw 2 , \
 "rocCurves/shenzenContROC.txt" using 2:1 title 'Shenzhen Uni' with lines lw 2 , \
 "rocCurves/OlaworksContROC.txt" using 2:1 title 'Olaworks' with lines lw 2 , \
 "rocCurves/illuxtechFrontalFaceContROC.txt" using 2:1 title 'Illuxtech Inc.' with lines lw 2
