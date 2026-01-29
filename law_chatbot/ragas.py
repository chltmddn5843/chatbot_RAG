from ragas import Dataset

def load_dataset():
    """Load test dataset for evaluation."""
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )

    data_samples = [
        {
            "question": "What is Ragas?",
            "grading_notes": "Ragas is an evaluation framework for LLM applications",
        },
        {
            "question": "How do metrics work?",
            "grading_notes": "Metrics evaluate the quality and performance of LLM responses",
        },
        # Add more test cases here
    ]

    for sample in data_samples:
        dataset.append(sample)

    dataset.save()
    return dataset


from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory

my_metric = DiscreteMetric(
    name="custom_evaluation",
    prompt="Evaluate this response: {response} based on: {context}. Return 'excellent', 'good', or 'poor'.",
    allowed_values=["excellent", "good", "poor"],
)

# ragas 평가 파이프라인 실행 함수 추가
from ragas.pipeline import RagasPipeline

def evaluate_with_ragas(dataset: Dataset):
    # LLM 모델 생성 - 실제 사용하시는 모델로 변경 가능
    llm = llm_factory(
        "mock",
        api_key=None,
    )

    # 평가 파이프라인 선언
    pipeline = RagasPipeline(
        dataset=dataset,
        llm=llm,
        metrics=[my_metric],
    )

    # 평가 실행
    results = pipeline.evaluate()

    # 평가 결과 출력
    for res in results:
        print(f"Question: {res['question']}")
        print(f"Grade: {res['custom_evaluation']}")
        print("---")

    return results


if __name__ == "__main__":
    ds = load_dataset()
    evaluate_with_ragas(ds)