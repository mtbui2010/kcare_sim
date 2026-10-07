# Edit the house

Everything about the apartment is in [`house_scene.py`](https://github.com/mtbui2010/kcare_sim/blob/main/sim/scripts/house_scene.py); the floor plan
it follows is `docs/draws/floor.drawio` (1 drawio unit = 2 cm).

| to change | where |
|---|---|
| rooms, walls, doors | `ROOMS`, `WALLS` — door gaps `(from, to)` along each wall; keep doors ≥ 0.9 m |
| a piece of furniture | its `F.append(body(...))` block in `furniture()` — position, yaw, boxes / cylinders / capsules |
| a place kcare can visit | `spot(key, room, centre, front_yaw, depth, width, height)` |
| objects on surfaces | `OBJECTS`: name, YCB id, spot, offset, yaw |
| the cabinet's place | `CAB_SITES` (`HOUSE_CAB_SITE=fridge` for the live sim) |
| textures | `sim/assets/real_home/*.png`, cut from photos |

After an edit:

```bash
python3 scripts/house_scene.py --render          # check top.png and the room views
python3 scripts/make_kcare_sim_location.py       # new places / poses for kcare
bash scripts/run_house.sh stop && bash scripts/run_house.sh start
python3 scripts/house_tour_test.py               # every place still reachable?
```

!!! warning "Training data depends on the room around the cabinet"
    The open-drawer data is rendered in the house. After changing anything the robot's cameras see
    from the cabinet, regenerate the dataset before training on it.
