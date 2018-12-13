LAZYFLOW_THREADS=32 LAZYFLOW_TOTAL_RAM_MB=450000 ~/ilastik-1.3.0-Linux/run_ilastik.sh --headless \
--project=/groups/scicompsoft/home/ackermand/Desktop/synapsePixelClassification.ilp \
--export_source="Simple Segmentation" \
--output_format="tiff sequence" \
--output_filename_format=/groups/scicompsoft/home/ackermand/lillvis/forIlastik/dataSubset3499_3998/Segmentation/{nickname}_{result_type}_{slice_index}.tiff \
"/groups/scicompsoft/home/ackermand/lillvis/forIlastik/dataSubset3499_3998/*.tif"
