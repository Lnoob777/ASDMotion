def route_video(duration, Tmin=2.0):
    """
    Decide routing based on video duration.
    """
    if duration >= Tmin:
        return "video"
    return "image"
