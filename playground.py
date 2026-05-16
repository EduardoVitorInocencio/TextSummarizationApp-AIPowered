from app.services.text_service import summarize_text
from app.services.sales_service import analyze_sales_data


def test_summary():

    text = """
    A inteligência artificial está revolucionando o mercado de tecnologia.
    Empresas utilizam IA para automatizar tarefas, reduzir custos e gerar insights.
    """

    result = summarize_text(
        text=text,
        provider="openai"
    )

    print("\n===== RESUMO =====\n")
    print(result)


def test_sales():

    result = analyze_sales_data(
        file_path="sales.xlsx",
        question="""
        Qual vendedor teve o melhor desempenho no ano de 2012
        e qual região teve maior faturamento?
        """,
        provider="openai"
    )

    print("\n===== INSIGHTS =====\n")
    print(result)


if __name__ == "__main__":

    # test_summary()

    test_sales()