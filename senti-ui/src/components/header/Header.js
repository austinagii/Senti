import Logo from "../logo/Logo";

import "./header.css";

export default function Header({ name }) {
  return (
    <>
      <section className="Header">
        <Logo className="Header-Logo"/>
        <p className="Header-UserText">Hey {name}</p>
      </section>
    </>
  );
}
