@echo off

cls

set PROCESS=prepare,alphapose,mediapipe,posetriplet,mix,motion
@REM set PROCESS=alphapose,mediapipe,posetriplet,mix,motion
@REM set PROCESS=mediapipe,posetriplet,mix,motion
@REM set PROCESS=posetriplet,mix,motion
@REM set PROCESS=motion
set LOGMODE=0
set LANG=ja
@REM set HAND=--hand-motion
set HAND=

python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\snobbism\snobbism.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%


@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\heart\heart_full5_mp4_20220929_141935" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\ivory\ivory_mp4_20220928_121801" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\bbf\bbf_mp4_20220928_105958" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\buster\buster_mp4_20220928_114021" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\night\night_mp4_20220929_151348" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\snobbism\snobbism_mp4_20220928_112703" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\suisei\suisei_mp4_20220929_160938" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\rocket\rocket_30fps_mp4_20221001_023830" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\seven\seven_mp4_20221001_032751" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\sugar\sugar_mp4_20221001_041756" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\galaxy\galaxy_mp4_20221001_051422" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\addiction\addiction_mp4_20221001_062355" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\charles\charles_mp4_20221001_075141" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\burai\B_mp4_20221001_020621" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\paranoia\paranoia_full_mp4_20220929_165134" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\yoiyoi\yoiyoi_mp4_20221005_073259" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\about\about_mp4_20221005_023105" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\check\check_mp4_20221005_031458" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\goast\goast_mp4_20221005_035446" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\green\green_mp4_20221005_044244" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\ikki\ikki_mp4_20221005_053414" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\ousama\ousama_mp4_20221005_062142" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\roki\roki_20191202_100912" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\teo\teo_mp4_20221005_065551" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\06\mahou\mahou_mp4_20221004_195838" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\06\taiyou\taiyou_mp4_20221005_014006" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\07\animaru\animaru_mp4_20221004_175155" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\07\alkali\alkali_mp4_20221004_152548" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\chika\chika_mp4_20221007_171241" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\cat\cat_30fps_mp4_20221007_192342" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\hisui\hisui_mp4_20221007_201402" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\kinyoubi\kinyoubi_full_mp4_20221007_205619" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\koisora\koisora_mp4_20221007_214823" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\sakura\sakura_mp4_20221007_172636" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\tugihagi\tugihagi_mp4_20221007_183608" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%




@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\02\buster\buster_0-1700.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --img-dir "E:\\MMD\\MikuMikuDance_v926x64\\Work\\201805_auto\\02\\buster\buster_0-1700_mp4_20221001_233113" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\bbf\bbf.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
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
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\03\haruhi\hare.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\07\alkali\alkali.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\07\animaru\animaru.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\06\mahou\mahou.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\06\taiyou\taiyou.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\about\about.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\check\check.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\goast\goast.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\green\green.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\ikki\ikki.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\ousama\ousama.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\roki\roki.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\05\teo\teo.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\04\yoiyoi\yoiyoi.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%

@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\chika\chika.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\sakura\sakura.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\tugihagi\tugihagi.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\cat\cat_30fps.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\hisui\hisui.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\kinyoubi\kinyoubi_full.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\koisora\koisora.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\kokoro\kokoro_full.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\midori\sm14811702.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\mirai\mirai.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\planetarium\planetarium.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
@REM python executor.py --video-file "E:\MMD\MikuMikuDance_v926x64\Work\201805_auto\01\revercible\revercible.mp4" --process %PROCESS% %HAND% --verbose 20 --log-mode %LOGMODE% --lang %LANG%
