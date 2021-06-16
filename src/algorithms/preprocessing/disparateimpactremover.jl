using Base: Symbol, String
""" 
    DisparateImpactRemoverWrapper

    Disparate impact remover is a preprocessing technique that edits feature
    values increase group fairness while preserving rank-ordering within groups
"""

struct DisparateImpactRemoverWrapper{M<:MLJBase.Model} <: DeterministicComposite
    grp::Symbol
    classifier::M
    repair_level::Float64
    sensitive_attribute::String
end

function BER_finder(X, y, privileged)
    labels = level(y)
    favLabel = labels[2]
    unfavLabel = labels[1]
    y_1 = y.==favLabel
    y_0 = y.==unfavLabel
    a = sum(y_0[privileged])
    c = sum(y_1[privileged])
    b = sum(y_0[.~privileged])
    d = sum(y_1[.~privileged])
    sensitivity = d/(b+d)
    specificity = a/(a+c)
    LRP = sensitivity/(1-specificity)
    BER =  (mean(y_0[privileged])+mean(y_1[.~privileged]))/2

end

function repairer(X, y)
    num_cols = size(X)[1]
    col_ids = 1:num_cols

    col_types = repeat(["Y"], length(col_ids))
    for i in col_ids
        if i in features_to_ignore
            col_types[i] = "I"
        elseif  i == self.feature_to_repair
            col_types[i] = "A"
        end
    end
        
    col_type_dict = Dict(temp[1]: temp[2] for temp in zip(col_ids, col_types))

    not_I_col_ids = [x for x in col_ids if col_type_dict[x] != "I"]

    if kdd
        cols_to_repair = [x for x in col_ids if col_type_dict[x] == "Y"]
    else
        cols_to_repair = [x for x in col_ids if col_type_dict[x] in ["Y","X"]]
    end

    safe_stratify_cols = [feature_to_repair]

    data_dict = Dict(col_id: [] for col_id in col_ids)

    for row in data_to_repair
        for i in col_ids
            append!(data_dict[i],row[i])
        end
    end

    # repair_types

    # for (col_id, values) in data_dict

    for col_id in not_I_col_ids:
        col_values = data_dict[col_id]
        col_values = sort!(col_values)
        unique_col_vals[col_id] = col_values
        index_lookup[col_id] = Dict(col_values[i]: i for i in 1:length(col_values))
    end
    
    unique_stratify_values = [unique_col_vals[i] for i in safe_stratify_cols]
    all_stratified_groups = Array{String}(undef, length(unique_stratify_values),length(unique_stratify_values))
    for i in length(unique_stratify_values)
        for j in length(unique_stratify_values)
            all_stratified_groups[i,j] = (unique_stratify_values[i],unique_stratify_values[j])
        end
    end
    #give list of values belonging to a stratified grp to be filled
    stratified_group_indices = 0

    val_sets = Dict(groups:Dict(col_id:Set() for col_id in cols_to_repair) for groups in all_stratified_groups)

    for (i, row) in enumerate(data_to_repair)
        group = tuple(row[col] for col in safe_stratify_cols)
        for col_id in cols_to_repair
            push!(val_sets[group][col_id],row[col_id])
        end
        push!(stratified_group_indices[group],i)

    end

    stratified_group_data = Dict(group: {} for group in all_stratified_groups)
    for group in all_stratified_groups:
        for (col_id, values) in data_dict
            indices = []
            for i in stratified_group_indices[group]
                value = col_dict[i]
                if value ∉ indices
                    indices[value] = []
                end
                push!(indices[value], i)
            end
            stratified_col_values = [(occurs, val) for (val, occurs) in indices]
            stratified_col_values = sort!(stratified_col_values, by = temp->temp[2])
            stratified_group_data[group][col_id] = stratified_col_values
        end
    end
    """Implement get mode"""
    mode_feature_to_repair = get_mode(data_dict[feature_to_repair])

    for col_id in cols_to_repair
        group_offsets = Dict(group:0 for group in all_stratified_groups)
        col = data_dict[col_id]

        num_quantiles = minimum!(length(val_sets[group][col_id]) for group in all_stratified_groups)
        quantile_unit = 1.0/num_quantiles

        if repair_types[col_id] in {int, float}
            for quantile in 1:num_quantiles
                median_at_quantiles = []
                indices_per_group = {}

                for group in all_stratified_groups
                    group_data_at_col = stratified_group_data[group][col_id]
                    num_val = length(group_data_at_col)
                    offset = convert(Int64, round(group_offsets[group]*num_val))
                    number_to_get = convert(Int64, round((group_offsets[group] + quantile_unit - offset))*num_val)
                    group_offsets[group] += quantile_unit

                    if number_to_get > 0

                        offset_data = group_data_at_col[offset:offset+number_to_get]
                        indices_per_group[group] = [i for (val_indices, _) in offset_data for i in val_indices]
                        values = sort!([convert(Float64,val) for (_, val) in offset_data])

                        push!(median_at_quantiles, get_median(values, kdd))
                    end
                end
            end
            median = get_median(median_at_quantiles, kdd)
            median_val_pos = index_lookup[col_id][median]

            for group in all_stratified_groups
                for index in indices_per_group[group]
                    original_value = col[index]
                    
                    current_val_pos = index_lookup[col_id][original_value]
                    distance = median_val_pos - current_val_pos
                    distance_to_repair =  convert(Int64, round(distance*repair_level))
                    index_of_repair_value = current_val_pos + distance_to_repair
                    repaired_value = unique_col_vals[col_id][index_of_repair_value]

                    data_dict[col_id][index] = repaired_value
                end
            end

        elseif repair_types[col_id] in {str}
            feature = CategoricalFeature(col)
            categories = feature.bin_index_dict.keys()
            group_features = get_group_data(all_stratified_groups, stratified_group_data, col_id)
            categories_count = get_categories_count(categories, all_stratified_groups, categories_count, group_features)
            categories_count_norm = get_categories_count_norm(categories, all_stratified_groups, categories_count, group_features)
            median = get_median_per_category(categories, categories_count_norm)
            # dist_generator Needs to be filled output
            # count_generator Needs to be filled out
            group_features, overflow = flow_on_group_features(all_stratified_groups, group_features, count_generator)
            group_features, assigned_overflow, distribution = assigned_overflow(all_stratified_groups, categories, overflow, group_features, dist_generator)

            for group in all_stratified_groups
                indices = stratified_group_indices[group]
                for (i, index) in enumerate(indices)
                    repaired_value = group_features[group][i] #may need correction
                    data_dict[col_id][index] = repaired_value
                end
            end
        end
        repaired_data = []
        for (i, orig_row) in enumerate(data_to_repair)
            new_row = [orig_row[j] if j ∉ cols_to_repair else data_dict[j][i] for j in col_ids]
            push!(repaired_data, new_row)
        end
    end
    return repaired_data
end

# function get_group_data(all_stratified_groups, stratified_group_data, col_id)
#     group_features={}
#     for group in all_stratified_groups:





        


            

# function checkDI(grps, y)