#!/usr/bin/env bash
# Download the YCB object scans the house uses (16 placed by default + the skill-bench extras; google_16k meshes +
# textures, ~450 MB) into sim/assets/ycb.
#   bash sim/scripts/fetch_ycb.sh
# Source: the YCB Object and Model Set (https://www.ycbbenchmarks.com), CC BY 4.0.
set -euo pipefail
DST="$(cd "$(dirname "$0")/.." && pwd)/assets/ycb"
URL=http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google
mkdir -p "$DST"
for id in 003_cracker_box 004_sugar_box 005_tomato_soup_can 006_mustard_bottle 011_banana 013_apple 017_orange \
          019_pitcher_base 021_bleach_cleanser 022_windex_bottle 024_bowl 025_mug 026_sponge 029_plate \
          056_tennis_ball 065-a_cups \
          002_master_chef_can 007_tuna_fish_can 008_pudding_box 009_gelatin_box 010_potted_meat_can 012_strawberry \
          014_lemon 015_peach 016_pear 018_plum 030_fork 031_spoon 040_large_marker 043_phillips_screwdriver \
          053_mini_soccer_ball 054_softball 055_baseball 057_racquetball 061_foam_brick 065-c_cups 065-f_cups \
          077_rubiks_cube; do
  [ -f "$DST/$id/google_16k/textured.obj" ] && continue
  echo "-> $id"
  curl -fsSL "$URL/${id}_google_16k.tgz" | tar xz -C "$DST"
done
echo "YCB objects in $DST"
