#!/bin/bash
# A second, independent house sim + kcare in the same container (parallel tests):
#   bash scripts/make_inst.sh inst2          -> /sim/inst2 (SIM_ROOT), shared code / assets / venvs by symlink,
#                                               own config, mjcf (house.xml + robot xml written by house_scene), out,
#                                               kcare_sim_configs.  Run it with ROS_DOMAIN_ID != 77 and its own port.
set -e
I=/sim/$1; mkdir -p $I/out/house $I/mjcf
for d in scripts assets .venv .kcare_venv third_party ros urdf data; do [ -e /sim/$d ] && ln -sfn /sim/$d $I/$d; done
[ -d $I/config ] || cp -r /sim/config $I/config
[ -d $I/kcare_sim_configs ] || cp -r /sim/kcare_sim_configs $I/kcare_sim_configs
for f in /sim/mjcf/*; do
  b=$(basename $f)
  case $b in house.xml|kaair_slamtec_mobile.xml) cp -f $f $I/mjcf/$b ;; *) ln -sfn $f $I/mjcf/$b ;; esac
done
cp -f /sim/out/house/occupancy.png $I/out/house/ 2>/dev/null || true
echo "$I ready"
