
def main():

    from wav2hyp import WAV2HYP
    from obspy import UTCDateTime

    # Initialize processor with configuration
    processor = WAV2HYP('./examples/sthelens.yaml')

    # Run processing
    catalog = processor.run(
        UTCDateTime('2004/10/05'),
        UTCDateTime('2004/10/06'),
        skip_picker=False,
        skip_associator=False,
        skip_locator=False,
    )

    print(f"Processed {len(catalog)} events")


if __name__ == "__main__":
    main()
