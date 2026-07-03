from .data import EarthquakeEncoding, StoreData

if __name__ == "__main__":
    dataset = StoreData(
        window_lags=60,
        output_lags=16,
        date_features=True,
        payday_features=True,
        earthquake_encoding=EarthquakeEncoding.DECAY,
    )
    _, n_stores, n_families = dataset.sales_tensor.shape
    print(dataset)
    print(f"n_input_channels={dataset.n_input_channels}")

    # Build your model here, then call run():
    #
    #   model = YourModel(
    #       n_input_channels=dataset.n_input_channels,
    #       n_stores=n_stores,
    #       n_families=n_families,
    #       n_output_steps=dataset.output_lags,
    #   )
    #   run(model, dataset, epochs=50, learning_rate=1e-3)
