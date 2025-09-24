SELECT 
    a.*
    , b.*
FROM api AS a
INNER JOIN {{ df_1 }} AS b
ON 1 = 1