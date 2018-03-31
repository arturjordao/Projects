grid on;
hold on;
max_pooling_global_avg = [0.9221666666666668 0.9195 0.9096666666666667 0.892 0.6471666666666668];
max_pooling_global_conf = [0.9256366934440139 0.9266874524929952 0.916168469627741 0.9011406366380502 0.659121873184166];
max_pooling_global_conf = max_pooling_global_conf-max_pooling_global_avg;

errorbar(max_pooling_global_avg, max_pooling_global_conf,'-ok');

% max_pooling8x8__avg = [0.9208333333333332 0.8966666666666667];
% max_pooling8x8_conf = [0.9291349793090454 0.9085953846712419];
% max_pooling8x8_conf = max_pooling8x8_conf-max_pooling8x8__avg;
% 
% errorbar(max_pooling8x8__avg, max_pooling8x8_conf,'-ok');



baseline_acc =[0.9241 0.9241 0.9241 0.9241 0.9241 0.9241];
baseline_std = [0.9273 0.9273 0.9273 0.9273 0.9273 0.9273];
baseline_std = baseline_std-baseline_acc;
errorbar(baseline_acc, baseline_std,'-or');


