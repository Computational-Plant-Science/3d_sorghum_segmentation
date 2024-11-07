# run pipeline
python3 /opt/code/ai_seg.py -i $INPUT -o $OUTPUT 

#singularity exec --nv docker://computationalplantscience/ai_u2net_color_clustering python3 /opt/code/ai_color_cluster_seg.py -p /groups/bucksch/test/30stops_ori/ -ft jpg -o /groups/bucksch/test/30stops_seg/ -s lab -c 2 -min 500 -max 1000000 -pl 0


# copy nested output files to working directory
find . -type f -name "*.jpg" -exec cp {} $WORKDIR \;


