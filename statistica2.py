import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Показать все столбцы без сокращений
pd.set_option('display.max_columns', None)

# Если текст в столбцах обрезается
pd.set_option('display.max_colwidth', None)

# Показать все строки (если нужно)
pd.set_option('display.max_rows', None)


df = pd.read_csv(r'....................')

print(df.head())

# Размер данных (строки, столбцы)
print(f"Размер данных: {df.shape}")

# Названия столбцов
print(f"Столбцы: {df.columns.tolist()}")

# Информация о данных (типы, количество не-null значений)
print(df.info())

# Типы данных каждого столбца
print(f"Типы данных:\n{df.dtypes}")

# Проверка пропущенных значений
print(f"Пропущенные значения:\n{df.isnull().sum()}")

# Если есть пропуски - удалить
df = df.dropna()

# Проверить, что пропуски удалены
print(f"Пропущенные значения после удаления:\n{df.isnull().sum()}")








# 2.	Постройте график распределения числа показов (Impressions) для каждой рекламы, прологарифмировав значения.
# Логарифмирую столбец Impressions
df['log_impressions'] = np.log1p(df['Impressions'])

# строим график
plt.figure(figsize=(12, 6))

# гистограмма
plt.subplot(1, 2, 1)
plt.hist(df['log_impressions'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('log(Impressions)')
plt.ylabel('Количество объявлений')
plt.title('Распределение log(Impressions) - гистограмма')
plt.grid(True, alpha=0.3)
plt.show()








print("\n" + "="*50 + "\n")
print("Создание колонки с CTR и анализ распределения\n")

# 3. Создайте новую колонку c CTR. Посмотрите на описательные статистики и распределение.

# Рассчитываем CTR для каждого объявления (в процентах)
# Добавляем проверку, чтобы избежать деления на ноль
df['CTR'] = (df['Clicks'] / df['Impressions']) * 100

# Проверим первые несколько строк, чтобы убедиться, что расчет верный
print("Первые 5 строк с новым столбцом CTR:")
print(df[['Impressions', 'Clicks', 'CTR']].head(10))
print("\n" + "="*50 + "\n")

# Описательные статистики для CTR
print("Описательные статистики для CTR (%):")
print(df['CTR'].describe())
print("\n" + "="*50 + "\n")

# Дополнительные статистики
print("Дополнительные статистики:")
print(f"Медиана CTR: {df['CTR'].median():.4f}%")
print(f"Мода CTR: {df['CTR'].mode().values[0]:.4f}%")
print(f"Дисперсия CTR: {df['CTR'].var():.4f}")
print(f"Стандартное отклонение CTR: {df['CTR'].std():.4f}")
print("\n" + "="*50 + "\n")

# Анализ распределения CTR
print("Анализ распределения CTR:")

# Проверим, есть ли объявления с нулевым CTR
zero_ctr = (df['CTR'] == 0).sum()
print(f"Количество объявлений с CTR = 0%: {zero_ctr} ({zero_ctr/len(df)*100:.2f}%)")

# Проверим максимальный CTR
max_ctr = df['CTR'].max()
print(f"Максимальный CTR: {max_ctr:.4f}%")







print("\n" + "="*50 + "\n")
print("Анализ CTR с разбивкой по рекламной кампании\n")

# 4. Проанализируйте CTR с разбивкой по рекламной кампании.

# Посмотрим, какие рекламные кампании есть в данных
print("Уникальные ID рекламных кампаний:")
print(df['xyz_campaign_id'].unique())
print(f"Всего кампаний: {df['xyz_campaign_id'].nunique()}")
print("\n" + "="*50 + "\n")

# Группировка по кампаниям и расчет статистик
campaign_stats = df.groupby('xyz_campaign_id').agg({
    'CTR': ['mean', 'median', 'std', 'min', 'max', 'count'],
    'Impressions': 'sum',
    'Clicks': 'sum'
}).round(4)

# Переименуем колонки для лучшей читаемости
campaign_stats.columns = ['CTR_mean', 'CTR_median', 'CTR_std', 'CTR_min', 'CTR_max', 'ad_count', 'total_impressions', 'total_clicks']
campaign_stats = campaign_stats.reset_index()

# Добавим общий CTR для кампании (сумма кликов / сумма показов * 100)
campaign_stats['campaign_CTR'] = (campaign_stats['total_clicks'] / campaign_stats['total_impressions'] * 100).round(4)

print("Статистики CTR по рекламным кампаниям:")
print(campaign_stats)
print("\n" + "="*50 + "\n")









print("\n" + "="*50 + "\n")
print("Расчет и анализ CPC (Cost Per Click)\n")

# 5. Посчитайте стоимость за клик пользователя по объявлению (CPC).
# Изучите полученные значения, используя меры центральной тенденции и меры изменчивости.

# Расчет CPC для каждого объявления
# CPC = spent / clicks (стоимость / количество кликов)
# Важно: обрабатываем случаи, где clicks = 0 (CPC будет бесконечным или неопределенным)

df['CPC'] = df['Spent'] / df['Clicks']

# Проанализируем проблему с нулевыми кликами
zero_clicks = df[df['Clicks'] == 0]
nonzero_clicks = df[df['Clicks'] > 0]

print(f"Всего объявлений: {len(df)}")
print(f"Объявлений с 0 кликов: {len(zero_clicks)} ({len(zero_clicks)/len(df)*100:.2f}%)")
print(f"Объявлений с >0 кликов: {len(nonzero_clicks)} ({len(nonzero_clicks)/len(df)*100:.2f}%)")

# Для анализа CPC будем использовать только объявления с ненулевыми кликами
df_cpc = df[df['Clicks'] > 0].copy()

print(f"\nДля анализа CPC используем {len(df_cpc)} объявлений с кликами")
print("\n" + "="*50 + "\n")

# МЕРЫ ЦЕНТРАЛЬНОЙ ТЕНДЕНЦИИ (Central Tendency)
print("МЕРЫ ЦЕНТРАЛЬНОЙ ТЕНДЕНЦИИ:")
print(f"Среднее арифметическое (mean): ${df_cpc['CPC'].mean():.4f}")
print(f"Медиана (median): ${df_cpc['CPC'].median():.4f}")
print(f"Мода (mode): ${df_cpc['CPC'].mode().values[0]:.4f}")

# Дополнительные меры центральной тенденции
from scipy import stats
print(f"Среднее геометрическое: ${stats.gmean(df_cpc['CPC']):.4f}")
print(f"Среднее гармоническое: ${stats.hmean(df_cpc['CPC']):.4f}")
print(f"Усеченное среднее (10%): ${stats.trim_mean(df_cpc['CPC'], 0.1):.4f}")

print("\n" + "="*50 + "\n")

# МЕРЫ ИЗМЕНЧИВОСТИ (Measures of Variability)
print("МЕРЫ ИЗМЕНЧИВОСТИ:")
print(f"Дисперсия (variance): ${df_cpc['CPC'].var():.6f}")
print(f"Стандартное отклонение (std): ${df_cpc['CPC'].std():.4f}")
print(f"Размах (range): ${df_cpc['CPC'].max() - df_cpc['CPC'].min():.4f}")
print(f"Межквартильный размах (IQR): ${df_cpc['CPC'].quantile(0.75) - df_cpc['CPC'].quantile(0.25):.4f}")
print(f"Коэффициент вариации (CV): {(df_cpc['CPC'].std() / df_cpc['CPC'].mean() * 100):.2f}%")

print("\n" + "="*50 + "\n")

# ПОЛНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ
print("ПОЛНЫЙ СТАТИСТИЧЕСКИЙ ОТЧЕТ CPC:")
print(df_cpc['CPC'].describe())

print("\n" + "="*50 + "\n")









print("\n" + "="*50 + "\n")
print("Визуализация CPC с разбивкой по полу пользователей\n")

# 6. Визуализируйте CPC с разбивкой по полу пользователей, которым были показаны объявления.

# Используем тот же датафрейм с ненулевыми кликами (df_cpc), который создали в шаге 5
# Но убедимся, что он существует
if 'df_cpc' not in locals():
    df_cpc = df[df['clicks'] > 0].copy()
    df_cpc['CPC'] = df_cpc['Spent'] / df_cpc['Clicks']

# Посмотрим распределение по полу в данных
print("Распределение объявлений по полу:")
print(df_cpc['gender'].value_counts())
print(f"\nВ процентах:")
print(df_cpc['gender'].value_counts(normalize=True) * 100)

print("\n" + "="*50 + "\n")

# Базовая статистика CPC по полу
print("СТАТИСТИКА CPC ПО ПОЛУ:")
gender_stats = df_cpc.groupby('gender')['CPC'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max',
    lambda x: x.quantile(0.25),
    lambda x: x.quantile(0.75)
]).round(4)

# Переименуем колонки для понятности
gender_stats.columns = ['Кол-во', 'Среднее', 'Медиана', 'Стд_откл', 'Мин', 'Макс', 'Q1', 'Q3']
print(gender_stats)

print("\n" + "="*50 + "\n")

# Дополнительные статистики по полу
print("ДОПОЛНИТЕЛЬНЫЕ СТАТИСТИКИ:")
for gender in df_cpc['gender'].unique():
    gender_data = df_cpc[df_cpc['gender'] == gender]['CPC']
    print(f"\n{gender}:")
    print(f"  Доля объявлений с CPC < $1: {(gender_data < 1).mean() * 100:.2f}%")
    print(f"  Доля объявлений с CPC > $5: {(gender_data > 5).mean() * 100:.2f}%")
    print(f"  Коэффициент вариации: {(gender_data.std() / gender_data.mean() * 100):.2f}%")
    print(f"  Асимметрия (skewness): {gender_data.skew():.4f}")
    print(f"  Эксцесс (kurtosis): {gender_data.kurtosis():.4f}")

print("\n" + "="*50 + "\n")

# Статистический тест: есть ли значимые различия в CPC между полами?
from scipy import stats

male_cpc = df_cpc[df_cpc['gender'] == 'M']['CPC']
female_cpc = df_cpc[df_cpc['gender'] == 'F']['CPC']

# t-тест для независимых выборок
t_stat, p_value = stats.ttest_ind(male_cpc, female_cpc, equal_var=False)

print("СТАТИСТИЧЕСКИЙ ТЕСТ (сравнение CPC мужчины vs женщины):")
print(f"T-статистика: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")

if p_value < 0.05:
    print("Вывод: Существуют статистически значимые различия в CPC между мужчинами и женщинами (p < 0.05)")
else:
    print("Вывод: Нет статистически значимых различий в CPC между мужчинами и женщинами (p >= 0.05)")

# Дополнительно: тест Манна-Уитни (непараметрический, если распределения не нормальны)
u_stat, p_value_mw = stats.mannwhitneyu(male_cpc, female_cpc)
print(f"\nТест Манна-Уитни (непараметрический):")
print(f"U-статистика: {u_stat:.4f}")
print(f"p-value: {p_value_mw:.6f}")

print("\n" + "="*50 + "\n")

# СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ
plt.figure(figsize=(18, 14))

# 1. Ящик с усами (boxplot) - основной график
plt.subplot(3, 3, 1)
sns.boxplot(data=df_cpc, x='gender', y='CPC')
plt.xlabel('Пол')
plt.ylabel('CPC ($)')
plt.title('Распределение CPC по полу\n(ящик с усами)')
plt.grid(True, alpha=0.3)
plt.show()







print("\n" + "="*50 + "\n")
print("Расчет конверсии из клика в покупку\n")

# 7. Посчитайте конверсию из клика в покупку.

df['purchase_conversion'] = (df['Approved_Conversion'] / df['Clicks']) * 100

# Анализ только для объявлений с кликами
df_conv = df[df['Clicks'] > 0].copy()

print(f"Всего объявлений: {len(df)}")
print(f"Объявлений с кликами: {len(df_conv)} ({len(df_conv)/len(df)*100:.1f}%)")
print(f"Объявлений без кликов: {len(df)-len(df_conv)} (конверсия не определена)")

print("\n" + "="*50 + "\n")
print("Статистики конверсии из клика в покупку (%):")
print(df_conv['purchase_conversion'].describe())

print("\n" + "="*50 + "\n")
print("Распределение конверсии:")

# Создадим категории конверсии
bins = [0, 1, 5, 10, 20, 50, 100]
labels = ['0-1%', '1-5%', '5-10%', '10-20%', '20-50%', '50-100%']
df_conv['conv_range'] = pd.cut(df_conv['purchase_conversion'], bins=bins, labels=labels)

# Вывод распределения
print(df_conv['conv_range'].value_counts().sort_index())

print("\n" + "="*50 + "\n")




