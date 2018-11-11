from utils import *

def generate_samples():
    df = read_file('data/cows_pregnancy_final.csv')
    cols = 9    
    x, y = get_samples_DeCock(df, cols)
    
    x.to_csv('data/cows_pregnancy_final_x.csv', index=False)
    y.to_csv('data/cows_pregnancy_final_y.csv', index=False)

    return True

def generate_vacadata():
    #df = join_by_index('Data/Vacadata1.csv', 'Data/Vacadata2.csv', 'ID')
    #join_to_csv(df, 'Data/Vacadata_1_2s.csv')
    df = read_file('Data/animales.csv')
    df = fix_missing_with_mode(df)
    df.loc[df['INTERPARTO'] >= 1000, 'INTERPARTO'] = 0
    df = drop_column(df, 'MADRE')

    #join_to_csv(df, 'Data/Vacadata_animal.csv')

    df_joined = join_by_index('Data/Vacadata_disease.csv', 'Data/Vacadata_Goal.csv', 'ID', 'inner')

    df_joined_2 = do_join(df, df_joined, 'ID', 'inner')

    df_dim = read_file('Data/Vacadata_DIM.csv')

    df_joined_3 = do_join(df_joined_2, df_dim, 'ID', 'inner')

    #join_to_csv(df_joined, 'joined_disease_goal2.csv')
    #join_to_csv(df_joined2, 'joined_pregnancy_dim.csv')
    
    df_dummies = dummies(df_joined_3, ['RAZA', 'PADRE'])
    print("NUMERO DE COLUMNAS")
    print(len(df.columns))
    join_to_csv(df_dummies, 'joined_final_dummies.csv')

if __name__ == '__main__':
    generate_vacadata()
    generate_samples()
