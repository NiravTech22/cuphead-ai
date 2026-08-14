"""knowledge/seed.py — built-in starter KB: ~30 well-documented public events + statements.

Lets the knowledge base work out of the box (no API key needed) and serves as
the acceptance fixture. Every entry below is a widely-documented, on-the-record
public statement with an existing film/audio/video recording (so the pipeline
can actually locate and fetch it) — no private individuals, no disputed or
unverifiable attributions. For depth, run ingest_events (your own structured
JSON) and ingest_subtitles (your own transcript library) afterwards.

    python -m app.knowledge.seed
"""
from __future__ import annotations

from ..logging_setup import get_logger
from . import db
from .embedder import embed, to_blob

log = get_logger(__name__)

# (title, year, type, genres/topic-tags, overview, popularity)
TITLES: list[tuple] = [
    ("I Have a Dream (March on Washington)", 1963, "speech", "Civil Rights, Oratory",
     "Martin Luther King Jr. addresses hundreds of thousands at the March on Washington for Jobs and Freedom from the steps of the Lincoln Memorial.", 96),
    ("FDR First Inaugural Address", 1933, "speech", "Politics, Great Depression",
     "Franklin D. Roosevelt is sworn in as president amid the Great Depression and rallies the nation against panic.", 88),
    ("FDR Pearl Harbor Address to Congress", 1941, "speech", "War, Politics",
     "Franklin D. Roosevelt asks Congress to declare war on Japan the day after the attack on Pearl Harbor.", 89),
    ("Churchill: We Shall Fight on the Beaches", 1940, "speech", "War, Oratory",
     "Winston Churchill addresses the House of Commons after the Dunkirk evacuation, vowing defiance against Nazi Germany.", 90),
    ("Churchill: Their Finest Hour", 1940, "speech", "War, Oratory",
     "Winston Churchill rallies Britain as the Battle of Britain looms.", 78),
    ("JFK Inaugural Address", 1961, "speech", "Politics, Oratory",
     "John F. Kennedy is sworn in as the 35th President and calls Americans to civic duty.", 93),
    ("MLK: I've Been to the Mountaintop", 1968, "speech", "Civil Rights",
     "Martin Luther King Jr.'s final speech, delivered the night before his assassination, in Memphis.", 80),
    ("Malcolm X: The Ballot or the Bullet", 1964, "speech", "Civil Rights",
     "Malcolm X lays out a choice between political and armed struggle for Black Americans.", 75),
    ("Nixon Checkers Speech", 1952, "speech", "Politics, Campaign",
     "Vice-presidential candidate Richard Nixon defends himself on television against corruption allegations.", 70),
    ("Eisenhower Farewell Address", 1961, "speech", "Politics",
     "Outgoing President Dwight D. Eisenhower warns of the growing power of the military-industrial complex.", 79),
    ("Nixon Resignation Address", 1974, "speech", "Politics, Watergate",
     "Richard Nixon announces his resignation from the presidency over the Watergate scandal.", 80),
    ("Nixon 'I Am Not a Crook' Press Conference", 1973, "press", "Politics, Watergate",
     "Richard Nixon defends his integrity to reporters during the Watergate investigation.", 76),
    ("Neil Armstrong Apollo 11 Moon Landing", 1969, "press", "Space, History",
     "Neil Armstrong becomes the first person to walk on the Moon during the Apollo 11 mission, broadcast live.", 92),
    ("Lou Gehrig Farewell Speech", 1939, "speech", "Sports",
     "New York Yankees first baseman Lou Gehrig addresses Yankee Stadium after being diagnosed with ALS.", 74),
    ("Muhammad Ali Press Remarks", 1964, "press", "Sports",
     "A young Cassius Clay, newly crowned heavyweight champion, tells reporters he is the greatest.", 77),
    ("Reagan: Tear Down This Wall (Brandenburg Gate)", 1987, "speech", "Cold War, Politics",
     "Ronald Reagan speaks at the Berlin Wall and challenges Soviet leader Mikhail Gorbachev.", 89),
    ("Reagan Challenger Disaster Address", 1986, "speech", "Space, Tragedy",
     "Ronald Reagan addresses the nation after the Space Shuttle Challenger disaster.", 80),
    ("George H.W. Bush: Read My Lips (1988 RNC)", 1988, "speech", "Politics, Campaign",
     "George H.W. Bush accepts the Republican presidential nomination and pledges not to raise taxes.", 77),
    ("Bill Clinton Grand Jury/Press Denial", 1998, "press", "Politics",
     "Bill Clinton denies allegations regarding Monica Lewinsky in a televised statement.", 75),
    ("Nelson Mandela Inauguration Speech", 1994, "speech", "Politics, South Africa",
     "Nelson Mandela is inaugurated as the first democratically elected President of South Africa.", 85),
    ("Mandela: I Am Prepared to Die (Rivonia Trial)", 1964, "speech", "Civil Rights, South Africa",
     "Nelson Mandela addresses the court from the dock at the Rivonia Trial.", 78),
    ("George W. Bush 9/11 Bullhorn Address", 2001, "speech", "Politics, 9/11",
     "George W. Bush addresses rescue workers at Ground Zero three days after the September 11 attacks.", 83),
    ("Obama New Hampshire Primary Speech", 2008, "speech", "Politics, Campaign",
     "Barack Obama addresses supporters after finishing second in the New Hampshire primary.", 84),
    ("Obama Election Night Victory Speech", 2008, "speech", "Politics",
     "Barack Obama addresses the nation in Grant Park, Chicago after winning the presidential election.", 86),
    ("Obama Charleston Eulogy (Amazing Grace)", 2015, "speech", "Politics",
     "Barack Obama delivers a eulogy for victims of the Charleston church shooting.", 76),
    ("Steve Jobs Stanford Commencement Address", 2005, "speech", "Technology, Commencement",
     "Apple co-founder Steve Jobs delivers a commencement address at Stanford University reflecting on life and death.", 90),
    ("Queen Elizabeth II COVID-19 Address", 2020, "speech", "Politics, COVID-19",
     "Queen Elizabeth II addresses the United Kingdom and Commonwealth during the COVID-19 pandemic.", 79),
    ("Greta Thunberg 'How Dare You' UN Speech", 2019, "speech", "Climate, UN",
     "Greta Thunberg addresses world leaders at the UN Climate Action Summit in New York.", 81),
    ("Barbara Jordan 1976 DNC Keynote", 1976, "speech", "Politics",
     "Barbara Jordan becomes the first Black woman to deliver a keynote address at a major-party convention.", 68),
    ("Ted Kennedy 1980 DNC Speech", 1980, "speech", "Politics",
     "Ted Kennedy addresses the Democratic National Convention after conceding the primary to Jimmy Carter.", 66),
]

# (title, quote, speaker, approx_timestamp_seconds or None)
QUOTES: list[tuple] = [
    ("I Have a Dream (March on Washington)",
     "I have a dream that my four little children will one day live in a nation where they will not be judged by the color of their skin but by the content of their character.",
     "Martin Luther King Jr.", None),
    ("I Have a Dream (March on Washington)",
     "Free at last! Free at last! Thank God almighty, we are free at last!",
     "Martin Luther King Jr.", None),
    ("I Have a Dream (March on Washington)",
     "I have a dream that one day this nation will rise up and live out the true meaning of its creed: we hold these truths to be self-evident, that all men are created equal.",
     "Martin Luther King Jr.", None),
    ("FDR First Inaugural Address", "The only thing we have to fear is fear itself.",
     "Franklin D. Roosevelt", None),
    ("FDR Pearl Harbor Address to Congress",
     "Yesterday, December 7th, 1941 — a date which will live in infamy — the United States of America was suddenly and deliberately attacked.",
     "Franklin D. Roosevelt", None),
    ("Churchill: We Shall Fight on the Beaches",
     "We shall fight on the beaches, we shall fight on the landing grounds, we shall fight in the fields and in the streets, we shall fight in the hills; we shall never surrender.",
     "Winston Churchill", None),
    ("Churchill: Their Finest Hour",
     "Let us therefore brace ourselves to our duties, and so bear ourselves that if the British Empire and its Commonwealth last for a thousand years, men will still say, 'This was their finest hour.'",
     "Winston Churchill", None),
    ("JFK Inaugural Address",
     "Ask not what your country can do for you — ask what you can do for your country.",
     "John F. Kennedy", None),
    ("JFK Inaugural Address",
     "Let the word go forth from this time and place, to friend and foe alike, that the torch has been passed to a new generation of Americans.",
     "John F. Kennedy", None),
    ("MLK: I've Been to the Mountaintop",
     "I've been to the mountaintop... I've seen the promised land. I may not get there with you, but I want you to know tonight that we as a people will get to the promised land.",
     "Martin Luther King Jr.", None),
    ("Malcolm X: The Ballot or the Bullet", "It'll be the ballot or it'll be the bullet.",
     "Malcolm X", None),
    ("Nixon Checkers Speech",
     "The kids, like all kids, love the dog, and I just want to say this right now, that regardless of what they say about it, we're gonna keep it.",
     "Richard Nixon", None),
    ("Eisenhower Farewell Address",
     "In the councils of government, we must guard against the acquisition of unwarranted influence... by the military-industrial complex.",
     "Dwight D. Eisenhower", None),
    ("Nixon Resignation Address", "I shall resign the presidency effective at noon tomorrow.",
     "Richard Nixon", None),
    ("Nixon 'I Am Not a Crook' Press Conference",
     "People have got to know whether or not their president is a crook. Well, I'm not a crook.",
     "Richard Nixon", None),
    ("Neil Armstrong Apollo 11 Moon Landing",
     "That's one small step for man, one giant leap for mankind.",
     "Neil Armstrong", None),
    ("Lou Gehrig Farewell Speech", "Today I consider myself the luckiest man on the face of the earth.",
     "Lou Gehrig", None),
    ("Muhammad Ali Press Remarks", "I am the greatest! I shook up the world!",
     "Muhammad Ali", None),
    ("Muhammad Ali Press Remarks", "Float like a butterfly, sting like a bee.",
     "Muhammad Ali", None),
    ("Reagan: Tear Down This Wall (Brandenburg Gate)", "Mr. Gorbachev, tear down this wall!",
     "Ronald Reagan", None),
    ("Reagan Challenger Disaster Address",
     "They slipped the surly bonds of earth to touch the face of God.",
     "Ronald Reagan", None),
    ("George H.W. Bush: Read My Lips (1988 RNC)", "Read my lips: no new taxes.",
     "George H.W. Bush", None),
    ("Bill Clinton Grand Jury/Press Denial",
     "I did not have sexual relations with that woman, Miss Lewinsky.",
     "Bill Clinton", None),
    ("Nelson Mandela Inauguration Speech",
     "Never, never and never again shall it be that this beautiful land will again experience the oppression of one by another.",
     "Nelson Mandela", None),
    ("Mandela: I Am Prepared to Die (Rivonia Trial)", "It is an ideal for which I am prepared to die.",
     "Nelson Mandela", None),
    ("George W. Bush 9/11 Bullhorn Address",
     "I can hear you. The rest of the world hears you. And the people who knocked these buildings down will hear all of us soon.",
     "George W. Bush", None),
    ("Obama New Hampshire Primary Speech", "Yes we can.", "Barack Obama", None),
    ("Obama Election Night Victory Speech",
     "If there is anyone out there who still doubts that America is a place where all things are possible... tonight is your answer.",
     "Barack Obama", None),
    ("Obama Charleston Eulogy (Amazing Grace)",
     "Amazing grace, how sweet the sound, that saved a wretch like me.",
     "Barack Obama", None),
    ("Steve Jobs Stanford Commencement Address", "Stay hungry, stay foolish.",
     "Steve Jobs", None),
    ("Steve Jobs Stanford Commencement Address",
     "Your time is limited, so don't waste it living someone else's life.",
     "Steve Jobs", None),
    ("Queen Elizabeth II COVID-19 Address",
     "We will be with our friends again; we will be with our families again; we will meet again.",
     "Queen Elizabeth II", None),
    ("Greta Thunberg 'How Dare You' UN Speech",
     "How dare you! You have stolen my dreams and my childhood with your empty words.",
     "Greta Thunberg", None),
    ("Barbara Jordan 1976 DNC Keynote",
     "My presence here is one additional bit of evidence that the American Dream need not forever be deferred.",
     "Barbara Jordan", None),
    ("Ted Kennedy 1980 DNC Speech",
     "For me, a few hours ago, this campaign came to an end... the work goes on, the cause endures, the hope still lives, and the dream shall never die.",
     "Ted Kennedy", None),
]


def run() -> None:
    conn = db.connect()
    try:
        title_ids: dict[str, int] = {}
        titles = [t for t in TITLES if t[1]]  # drop placeholder rows
        for title, year, type_, genres, overview, pop in titles:
            title_ids[title] = db.upsert_title(
                conn, title=title, year=year, type_=type_, genres=genres,
                overview=overview, popularity=pop)

        # wipe + reinsert seed quotes for idempotency
        seed_tids = tuple(title_ids.values())
        conn.execute(f"DELETE FROM quotes WHERE title_id IN ({','.join('?'*len(seed_tids))})",
                     seed_tids)
        conn.execute("INSERT INTO quotes_fts(quotes_fts) VALUES('rebuild')")
        quote_rows: list[tuple[int, str]] = []
        for title, quote, speaker, ts in QUOTES:
            if title not in title_ids:
                continue
            qid = db.add_quote(conn, title_ids[title], quote, speaker, ts)
            quote_rows.append((qid, quote))

        # embed everything (CPU)
        overviews = [f"{t[0]} ({t[1]}). {t[3]}. {t[4]}" for t in titles]
        tvecs = embed(overviews)
        db.put_embeddings(conn, "titles", [
            (title_ids[t[0]], to_blob(v)) for t, v in zip(titles, tvecs)])
        qvecs = embed([q for _, q in quote_rows])
        db.put_embeddings(conn, "quotes", [
            (qid, to_blob(v)) for (qid, _), v in zip(quote_rows, qvecs)])
        conn.commit()
        log.info("seeded KB: %s", db.stats(conn))
    finally:
        conn.close()


if __name__ == "__main__":
    run()
