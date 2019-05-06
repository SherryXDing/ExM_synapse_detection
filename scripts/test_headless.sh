# # Object classification
# LAZYFLOW_THREADS=32 LAZYFLOW_TOTAL_RAM_MB=45000 \
# /groups/scicompsoft/home/dingx/Apps/ilastik-1.3.0-Linux/run_ilastik.sh --headless \
# --project=/groups/scicompsoft/home/dingx/Documents/ExM/model_ilastik/Obj_Classify_0723.ilp \
# --export_source="Object Predictions" \
# --raw_data=/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Data_16bit/Test_BGsubtract_radius20.tif --output_format="multipage tiff" \
# --output_filename_format=/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Output_16bit/Test.tiff \
# --table_filename=/groups/scicompsoft/home/dingx/Documents/ExM/Ch0/Output_16bit/Test_obj_features_model0723.csv \
# --pipeline_result_drange="(0,2)" --export_drange="(0,255)" --export_dtype="uint16" --output_axis_order="tzxyc"


# Pixel classification
/groups/scicompsoft/home/dingx/Apps/ilastik-1.3.0-Linux/run_ilastik.sh --headless \
--project=/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_ilastik/model_all_crops/model_pxl_all_crop.ilp \
--export_source="Simple Segmentation" \
--output_format="multipage tiff" \
--pipeline_result_drange="(0,1)" --export_drange="(0,1)" \
--output_filename_format=/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/model_ilastik/model_all_crops/optic_lobe_1.tiff \
"/groups/dickson/dicksonlab/lillvis/ExM/Ding-Ackerman/crops-for-training_Oct2018/DING/data/optic_lobe_1/C1-4228_3823_4701-background_subtract.tif"
# --export_source="Simple Segmentation" or "Probabilities"\