def _remove_inst(inst_map, remove_id_list):
    """Remove instances with id in remove_id_list.

    Args:
        inst_map: map of instances
        remove_id_list: list of ids to remove from inst_map
    """
    for inst_id in remove_id_list:
        inst_map[inst_map == inst_id] = 0
    return inst_map


####
def _get_patch_top_left_info(img_shape, input_size, output_size):
    """Get top left coordinate information of patches from original image.

    Args:
        img_shape: input image shape
        input_size: patch input shape
        output_size: patch output shape

    """
    in_out_diff = input_size - output_size
    nr_step = np.floor((img_shape - in_out_diff) / output_size) + 1
    last_output_coord = (in_out_diff // 2) + (nr_step) * output_size
    # generating subpatches index from orginal
    output_tl_y_list = np.arange(
        in_out_diff[0] // 2, last_output_coord[0], output_size[0], dtype=np.int32
    )
    output_tl_x_list = np.arange(
        in_out_diff[1] // 2, last_output_coord[1], output_size[1], dtype=np.int32
    )
    output_tl_y_list, output_tl_x_list = np.meshgrid(output_tl_y_list, output_tl_x_list)
    output_tl = np.stack(
        [output_tl_y_list.flatten(), output_tl_x_list.flatten()], axis=-1
    )
    input_tl = output_tl - in_out_diff // 2
    return input_tl, output_tl


#### all must be np.array
def _get_tile_info(img_shape, tile_shape, ambiguous_size=128):
    """Get information of tiles used for post processing.

    Args:
        img_shape: input image shape
        tile_shape: tile shape used for post processing
        ambiguous_size: used to define area at tile boundaries

    """
    # * get normal tiling set
    tile_grid_top_left, _ = _get_patch_top_left_info(img_shape, tile_shape, tile_shape)
    tile_grid_bot_right = []
    for idx in list(range(tile_grid_top_left.shape[0])):
        tile_tl = tile_grid_top_left[idx][:2]
        tile_br = tile_tl + tile_shape
        axis_sel = tile_br > img_shape
        tile_br[axis_sel] = img_shape[axis_sel]
        tile_grid_bot_right.append(tile_br)
    tile_grid_bot_right = np.array(tile_grid_bot_right)
    tile_grid = np.stack([tile_grid_top_left, tile_grid_bot_right], axis=1)
    tile_grid_x = np.unique(tile_grid_top_left[:, 1])
    tile_grid_y = np.unique(tile_grid_top_left[:, 0])
    # * get tiling set to fix vertical and horizontal boundary between tiles
    # for sanity, expand at boundary `ambiguous_size` to both side vertical and horizontal
    stack_coord = lambda x: np.stack([x[0].flatten(), x[1].flatten()], axis=-1)
    tile_boundary_x_top_left = np.meshgrid(
        tile_grid_y, tile_grid_x[1:] - ambiguous_size
    )
    tile_boundary_x_bot_right = np.meshgrid(
        tile_grid_y + tile_shape[0], tile_grid_x[1:] + ambiguous_size
    )
    tile_boundary_x_top_left = stack_coord(tile_boundary_x_top_left)
    tile_boundary_x_bot_right = stack_coord(tile_boundary_x_bot_right)
    tile_boundary_x = np.stack(
        [tile_boundary_x_top_left, tile_boundary_x_bot_right], axis=1
    )
    #
    tile_boundary_y_top_left = np.meshgrid(
        tile_grid_y[1:] - ambiguous_size, tile_grid_x
    )
    tile_boundary_y_bot_right = np.meshgrid(
        tile_grid_y[1:] + ambiguous_size, tile_grid_x + tile_shape[1]
    )
    tile_boundary_y_top_left = stack_coord(tile_boundary_y_top_left)
    tile_boundary_y_bot_right = stack_coord(tile_boundary_y_bot_right)
    tile_boundary_y = np.stack(
        [tile_boundary_y_top_left, tile_boundary_y_bot_right], axis=1
    )
    tile_boundary = np.concatenate([tile_boundary_x, tile_boundary_y], axis=0)
    # * get tiling set to fix the intersection of 4 tiles
    tile_cross_top_left = np.meshgrid(
        tile_grid_y[1:] - 2 * ambiguous_size, tile_grid_x[1:] - 2 * ambiguous_size
    )
    tile_cross_bot_right = np.meshgrid(
        tile_grid_y[1:] + 2 * ambiguous_size, tile_grid_x[1:] + 2 * ambiguous_size
    )
    tile_cross_top_left = stack_coord(tile_cross_top_left)
    tile_cross_bot_right = stack_coord(tile_cross_bot_right)
    tile_cross = np.stack([tile_cross_top_left, tile_cross_bot_right], axis=1)
    return tile_grid, tile_boundary, tile_cross


####
def _get_chunk_patch_info(
        img_shape, chunk_input_shape, patch_input_shape, patch_output_shape
):
    """Get chunk patch info. Here, chunk refers to tiles used during inference.

    Args:
        img_shape: input image shape
        chunk_input_shape: shape of tiles used for post processing
        patch_input_shape: input patch shape
        patch_output_shape: output patch shape

    """
    round_to_multiple = lambda x, y: np.floor(x / y) * y
    patch_diff_shape = patch_input_shape - patch_output_shape

    chunk_output_shape = chunk_input_shape - patch_diff_shape
    chunk_output_shape = round_to_multiple(
        chunk_output_shape, patch_output_shape
    ).astype(np.int64)
    chunk_input_shape = (chunk_output_shape + patch_diff_shape).astype(np.int64)

    patch_input_tl_list, _ = _get_patch_top_left_info(
        img_shape, patch_input_shape, patch_output_shape
    )
    patch_input_br_list = patch_input_tl_list + patch_input_shape
    patch_output_tl_list = patch_input_tl_list + patch_diff_shape
    patch_output_br_list = patch_output_tl_list + patch_output_shape
    patch_info_list = np.stack(
        [
            np.stack([patch_input_tl_list, patch_input_br_list], axis=1),
            np.stack([patch_output_tl_list, patch_output_br_list], axis=1),
        ],
        axis=1,
    )

    chunk_input_tl_list, _ = _get_patch_top_left_info(
        img_shape, chunk_input_shape, chunk_output_shape
    )
    chunk_input_br_list = chunk_input_tl_list + chunk_input_shape
    # * correct the coord so it stay within source image
    y_sel = np.nonzero(chunk_input_br_list[:, 0] > img_shape[0])[0]
    x_sel = np.nonzero(chunk_input_br_list[:, 1] > img_shape[1])[0]
    chunk_input_br_list[y_sel, 0] = (
                                            img_shape[0] - patch_diff_shape[0]
                                    ) - chunk_input_tl_list[y_sel, 0]
    chunk_input_br_list[x_sel, 1] = (
                                            img_shape[1] - patch_diff_shape[1]
                                    ) - chunk_input_tl_list[x_sel, 1]
    chunk_input_br_list[y_sel, 0] = round_to_multiple(
        chunk_input_br_list[y_sel, 0], patch_output_shape[0]
    )
    chunk_input_br_list[x_sel, 1] = round_to_multiple(
        chunk_input_br_list[x_sel, 1], patch_output_shape[1]
    )
    chunk_input_br_list[y_sel, 0] += chunk_input_tl_list[y_sel, 0] + patch_diff_shape[0]
    chunk_input_br_list[x_sel, 1] += chunk_input_tl_list[x_sel, 1] + patch_diff_shape[1]
    chunk_output_tl_list = chunk_input_tl_list + patch_diff_shape // 2
    chunk_output_br_list = chunk_input_br_list - patch_diff_shape // 2  # may off pixels
    chunk_info_list = np.stack(
        [
            np.stack([chunk_input_tl_list, chunk_input_br_list], axis=1),
            np.stack([chunk_output_tl_list, chunk_output_br_list], axis=1),
        ],
        axis=1,
    )

    return chunk_info_list, patch_info_list

