class DayHour:
    @staticmethod
    def from_string(time_str):
        try:
            day, hour, minutes = map(int, time_str.split(":"))
            return DayHour(day, hour + minutes / 60)
        except ValueError:
            raise ValueError("Invalid time format. Use 'day:hour:minutes'.")

    def __init__(self, day: int = 0, hour: float = 0, minutes: float = 0):
        if not (0 <= hour + minutes / 60 < 24):
            raise ValueError("Time must be between 0 and 23:59.")
        if not (0 <= minutes < 60):
            raise ValueError("Minutes must be between 0 and 59.")
        self.day: int = day
        self.hour: float = hour + minutes / 60

    def __repr__(self):
        return f"DayHour({self.day}, {self.hour})"

    def to_tuple(self):
        return (self.day, self.hour)

    def __eq__(self, other):
        if isinstance(other, DayHour):
            return self.day == other.day and self.hour == other.hour
        return False

    def __lt__(self, other):
        if isinstance(other, DayHour):
            return (self.day, self.hour) < (other.day, other.hour)
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, DayHour):
            return (self.day, self.hour) <= (other.day, other.hour)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, DayHour):
            total_hours = self.day * 24 + self.hour - other.day * 24 - other.hour
            new_day, new_hour = divmod(total_hours, 24)
            return DayHour(new_day, new_hour)  # type: ignore

        if isinstance(other, Duration):
            total_hours = self.hour - other.hours
            extra_days, new_hour = divmod(total_hours, 24)
            new_day = self.day + extra_days
            if new_hour < 0:
                new_hour += 24
                new_day -= 1
            return DayHour(new_day, new_hour)  # type: ignore
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, Duration):
            total_hours = self.hour + other.hours
            extra_days, new_hour = divmod(total_hours, 24)
            new_day = self.day + extra_days
            return DayHour(new_day, new_hour)  # type: ignore
        return NotImplemented


class Duration:
    @staticmethod
    def from_string(duration_str):
        try:
            hours, minutes = map(int, duration_str.split(":"))
            return Duration(hours + minutes / 60)
        except ValueError:
            raise ValueError("Invalid duration format. Use 'hours:minutes'.")

    def __init__(self, hours: float, minutes: float = 0):
        if not (0 <= minutes < 60):
            raise ValueError("Minutes must be between 0 and 59.")
        if not (0 <= hours < 24):
            raise ValueError("Hours must be between 0 and 23.")
        self.hours: float = hours + minutes / 60

    def __repr__(self):
        return f"Duration(hours={self.hours})"

    def __eq__(self, other):
        if isinstance(other, Duration):
            return self.hours == other.hours
        return False

    def __le__(self, other):
        if isinstance(other, Duration):
            return self.hours <= other.hours
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Duration(self.hours * other)
        return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, DayHour):
            total_hours = self.hours + other.hour
            extra_days, new_hour = divmod(total_hours, 24)
            new_day = other.day + int(extra_days)
            return DayHour(new_day, new_hour)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Duration):
            other_time = other.hours
            self_time = self.hours
            return self_time / other_time
        return NotImplemented
