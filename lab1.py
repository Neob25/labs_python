import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%config InlineBackend.figure_format = 'svg'
pd.set_option("display.precision", 2)

data = pd.read_csv('titanic_train.csv', index_col='PassengerId')


#кол-во муж и жен
sex_counts = data['Sex'].value_counts()
num_male = sex_counts.get('male', 0)
num_female = sex_counts.get('female', 0)
print(f"м: {num_male}, ж: {num_female}")

#люди второго класса
pclass_counts = data['Pclass'].value_counts().sort_index()

pclass_by_sex = data.groupby(['Sex', 'Pclass']).size().unstack()

total = pclass_counts.loc[2]
male_in = pclass_by_sex.loc['male', 2]
female_in = pclass_by_sex.loc['female', 2]

print(f"\nВсего пассажиров во 2-м классе: {total}")
print(f"Мужчин во 2-м классе: {male_in}")
print(f"Женщин во 2-м классе: {female_in}")
print()


#медиана и стнд отклонение
median = data['Fare'].median()
fare_std = data['Fare'].std()
print(f"медиана: {median:.2f}")
print(f"стнд отклонение: {fare_std:.2f}")
print()

#средний возраст выжившего
survival = data.groupby('Survived')['Age'].mean()
print(survival)

dead = survival.loc[0]
survived = survival.loc[1]

print(f" сред возр умерших: {dead:.2f}")
print(f" сред возр выживших: {survived:.2f}")
print("выше ли возраст:", survived > dead)
print()

#выжившие среди молодых и пожилых
young = data['Age'] < 30
old = data['Age'] > 60

print("<30:", young.sum())
print(">60:", old.sum())

young_survival = data.loc[young, 'Survived'].mean()
old_survival = data.loc[old, 'Survived'].mean()

print(f" выжившие <30: {young_survival*100:.1f}%")
print(f" выжившие >60: {old_survival*100:.1f}%")
print()

#чаще выживали женщины?
survival_by_sex = data.groupby('Sex')['Survived'].mean()
print(survival_by_sex)
print(f" выжившие  мужчины:  {survival_by_sex.loc['male']*100:.1f}%")
print(f" выжившие  женщины: {survival_by_sex.loc['female']*100:.1f}%")
print()

#популярное имя
male_data = data[data['Sex'] == 'male'].copy()

def extract_real_name(full_name):
    try:
        parts = full_name.split(', ')
        if len(parts) < 2:
            return full_name
            
        name_part = parts[1]
        words = name_part.split()
        
        real_names = []
        for word in words:
            if '.' in word:
                continue
            if word.strip():
                real_names.append(word)
        
        if real_names:
            return real_names[0]
        else:
            return name_part.split()[0] if name_part.split() else full_name
            
    except:
        return full_name

male_data['RealName'] = male_data['Name'].apply(extract_real_name)
name_counts = male_data['RealName'].value_counts()

most_common_name = name_counts.index[0]
print(f"наиболее популярное имя: {most_common_name}")
print()


#возраста по полу и классу
age_stats = data.groupby(['Sex', 'Pclass'])['Age'].mean().unstack()

if age_stats.loc['male', 1] > 40:
    print("в среднем мужчины 1 класса старше 40 лет")
    
if age_stats.loc['female', 1] > 40:
    print("в среднем женщины 1 класса старше 40 лет")

if all(age_stats.loc['male', c] > age_stats.loc['female', c] for c in [1, 2, 3]):
    print("мужчины всех классов в среднем старше, чем женщины того же класса")

if all(age_stats.loc[s, 1] > age_stats.loc[s, 2] > age_stats.loc[s, 3] for s in ['male', 'female']):
    print("пассажиры 1 класса старше 2-го класса, которые старше 3-го класса")
