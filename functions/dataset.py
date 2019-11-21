
#Read dataset
def read_dataset(path,pd):
    data = pd.read_csv(path)
    pd.set_option('display.max_columns', 999) #shows all cols
    return data


#Check null values
def print_details(df):
    # print(df.isna().sum())
    print(df.head())
    print(df.shape)

#Select data
def select_data(data):
    df = data[['court_surface', 'prize_money', 'year', 'player_id', 'opponent_id', 'tournament', 'num_sets', 'sets_won', 'games_won', 'games_against', 'nation']]
    return df

#Drop na
def drop_na_values(df):
    df.dropna(subset = ['court_surface', 'prize_money', 'year', 'player_id', 'opponent_id', 'tournament', 'num_sets', 'sets_won', 'games_won', 'games_against', 'nation'],inplace = True)
    df.reset_index(drop=True, inplace=True)
