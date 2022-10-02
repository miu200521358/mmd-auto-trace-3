@echo off

cls

@REM set PROCESS=prepare,alphapose,mediapipe,posetriplet,mix,motion
set PROCESS=alphapose,mediapipe,posetriplet,mix,motion
@REM set PROCESS=mediapipe,posetriplet,mix,motion
@REM set PROCESS=mix,motion
@REM set PROCESS=motion
set LOGMODE=0
set LANG=ja
@REM set HAND=--hand-motion
set HAND=

@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\buster\buster_0-1700.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\\MMD\\MikuMikuDance_v926x64\\Work\\201805_auto\\02\\buster\buster_0-1700_mp4_20221001_233113" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\buster\buster_mp4_20220928_114021" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\bbf\bbf_mp4_20220928_105958" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\snobbism\snobbism_mp4_20220928_112703" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\ivory\ivory_mp4_20220928_121801" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\heart\heart_full5_mp4_20220929_141935" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\night\night_mp4_20220929_151348" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\suisei\suisei_mp4_20220929_160938" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\paranoia\paranoia_full_mp4_20220929_165134" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\burai\B_mp4_20221001_020621" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\rocket\rocket_30fps_mp4_20221001_023830" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\seven\seven_mp4_20221001_032751" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\sugar\sugar_mp4_20221001_041756" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\galaxy\galaxy_mp4_20221001_051422" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\addiction\addiction_mp4_20221001_062355" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\charles\charles_mp4_20221001_075141" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%



@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\bbf\bbf.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\snobbism\snobbism.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\buster\buster.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\ivory\ivory.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\heart\heart_full5.mp4 --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\night\night.mp4 --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\suisei\suisei.mp4 --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\paranoia\paranoia_full.mp4 --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\burai\B.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\rocket\rocket_30fps.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\seven\seven.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\sugar\sugar.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\galaxy\galaxy.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\addiction\addiction.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\charles\charles.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
