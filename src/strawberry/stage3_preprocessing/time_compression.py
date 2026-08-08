from datetime import date, datetime, time, timedelta


RECORDING_START = time(8, 0, 0)
RECORDING_END = time(20, 0, 0)
SKIPPED_RECORDING_DATES = {
    date(2026, 3, 22),
    date(2026, 3, 29),
}


def compress_recording_timestamp(timestamp: datetime) -> datetime:
    """Map daytime-only recordings onto a continuous 12-hour recording timeline."""
    recording_day = timestamp.date()
    if recording_day in SKIPPED_RECORDING_DATES:
        raise ValueError(f"Cannot compress skipped recording date: {recording_day}")

    anchor = datetime.combine(date(2026, 3, 18), RECORDING_START)
    day_index = _recording_day_index(recording_day)
    day_offset = datetime.combine(timestamp.date(), timestamp.time()) - datetime.combine(recording_day, RECORDING_START)

    if day_offset < timedelta(0):
        day_offset = timedelta(0)

    block_duration = _recording_block_duration()
    return anchor + (day_index * block_duration) + day_offset


def _recording_day_index(recording_day: date) -> int:
    start_day = date(2026, 3, 18)
    if recording_day < start_day:
        raise ValueError(f"Recording date is before timeline start: {recording_day}")

    index = 0
    current = start_day
    while current < recording_day:
        if current not in SKIPPED_RECORDING_DATES:
            index += 1
        current += timedelta(days=1)

    return index


def _recording_block_duration() -> timedelta:
    return datetime.combine(date.min, RECORDING_END) - datetime.combine(date.min, RECORDING_START)
