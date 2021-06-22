SELECT field_data_body.entity_id,
       field_data_body.body_value,
       db_summary.field_summary_value,
       db_category.name_category,
       db_vakgebied.name_vakgebied
    FROM field_data_body
            LEFT JOIN
(SELECT entity_id as ent_id2, field_summary_value
    FROM field_data_field_summary) as db_summary
        ON field_data_body.entity_id = db_summary.ent_id2
            LEFT JOIN
(SELECT entity_id as ent_id3, field_category_tid, taxonomy.name as name_category
    FROM field_data_field_category
        LEFT JOIN
        (SELECT tid, name FROM taxonomy_term_data) as taxonomy
        ON field_category_tid=taxonomy.tid
    ) as db_category
        ON field_data_body.entity_id = db_category.ent_id3
            LEFT JOIN
(SELECT entity_id as ent_id4, field_vakgebied_tid, taxonomy.name name_vakgebied
    FROM field_data_field_vakgebied
        LEFT JOIN
        (SELECT tid, name FROM taxonomy_term_data) as taxonomy
        ON field_vakgebied_tid=taxonomy.tid
    ) as db_vakgebied
        ON field_data_body.entity_id = db_vakgebied.ent_id4
